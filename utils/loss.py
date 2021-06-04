# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        # 获取设备
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 标签平滑，eps默认为0，其实就是没用
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        # 如果设置 'fl_gamma' 就是好用focal loss，默认不使用
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)


        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        '''
        p:网络输出 [batch_size, num_anchors, h, w, bbox + 1 + cls] * 3
        p[0] = 1x3x80x80x85
        p[1] = 1x3x40x40x85
        p[2] = 1x3x20x20x85

        target: labels [num_labels, image_index + cls_id + bbox]
        '''
        device = targets.device

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls:  类别， tbox： 坐标偏移量和wh， indices: image, anchor, grid indices ; anchors :锚框
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # *2 为了进行对靠近网络边界点的目标进行检测
                pxy = ps[:, :2].sigmoid() * 2. - 0.5 # 预测框 bbox 中心点位置
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # 预测框的hw
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # Ciou = true ，计算的是Ciou损失
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # 根据model.gr 设置objectness的标签值
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # if 类别数大于1 才计算分类损失。
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参数设置的各个部分损失的系数，获取最终损失。
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """
        p: 网络输出[batch_size, num_anchors, h, w, nc+5(bbox + 1(confidence) + cls)] * 3( 3 个特征图)
        target: labels [num_labels, image_index + cls_id + bbox(xywh)]， targets.shape = [nt, 6] icxywh
        model: yolov5

        这个函数用来构建计算损失的目标值
        1.先找出哪些真实框是由哪些锚框负责预测的， 给相应的真实框添加锚框索引
        2.再找出哪些网格是负责预测的，例如label中的中心点坐标在网格中偏左下方，就由当前网格和他的左侧和下方网格一起用来预测目标
        3.根据1和2将结果存储在变量tcls, tbox, indices, anch中
        """
        na, nt = self.na, targets.shape[0]  # number of anchors（3）, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # 生成 anchor 索引 ai.shape = [na, nt]
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  
        '''
        na * nt
        ai = [
            [0,0,0,0,.....0]
            [1,1,1,1,.....1]
            [2,2,2,2,.....2]
        ]
        '''
        # same as .repeat_interleave(nt)
        # targets.shape = [na, nt ,7]
        '''
        target1 = targets.repeat(na, 1, 1) # target.shape = [3, nt, 6]
        targets = torch.cat((target1 , ai[:, :, None]), 2)
        [na, nt, 7]
        [img_idx, cla_id, x, y, w, h， anchor_idx(0, 1, 2)]

        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 设置偏移量
        g = 0.5  # bias
        # (5, 2)
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对每个检测层进行处理。
        for i in range(self.nl):
            # self.anchors.shape = [3,3,2], yaml 中的ahchors view(3, -1, 2)
            anchors = self.anchors[i]
            # p[i].shape = (b, c, h, w, nc + 5), 
            # gain = [1, 1, w, h, w, h, 1]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将标签框的xywh从基于0-1, 映射到 基于特征图
            # [na, nt, 7] * [7, ]
            t = targets * gain
            if nt:
                # Matches
                """
                预测的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的(这应该是yolov5的创新点)，
                从而更好的回归(与新的边框回归方式有关)
                由于yolov3回归wh采用的是out=exp(in)，这很危险，因为out=exp(in)可能会无穷大，就会导致失控的梯度，不稳定，
                NaN损失并最终完全失去训练；
                (当然原yolov3采用的是将targets进行反算来求in与网络输出的结果，就问题不大，但采用iou loss，
                就需要将网络输出算成out来进行loss求解，所以会面临这个问题)；
                所以作者采用新的wh回归方式:
                (wh.sigmoid() * 2) ** 2 * anchors[i], 原来yolov3为anchors[i] * exp(wh)
                将标签框与anchor的倍数控制在0~4之间；
                hyp.scratch.yaml中的超参数anchor_t=4，所以也是通过此参数来判定anchors与标签框契合度；
                """
                # 计算宽高比值，特征图上的标签宽高 与 anchors的宽高 之比
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                """
                筛选满足1 / hyp['anchor_t'] < targets_wh/anchor_wh < hyp['anchor_t']的框;
                由于wh回归公式中将标签框与anchor的倍数控制在0~4之间，所以这样筛选之后也会浪费一些输出空间；
                由于分给每个特征金字塔层的anchor尺度都不一样，这里根据标签wh与anchor的wh的比例分配标签，
                就相当于把不同尺度的GT分配给了不同的特征层来回归；
                """
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 筛选后的t.shape = [M, 7], M 为筛选后的数量。
                t = t[j]  # filter

                # Offsets
                # 找到这是框的中心点，再添加两个距离它最近的两个点也作为正样本
                # 参考博文https://blog.csdn.net/cdknight_happy/article/details/109817548#t11
                # x小于0.5就靠近左边的网格，y小于0.5就靠近上边的网格
                gxy = t[:, 2:4]  # grid xy  真实框中心坐标x, y， 相对于图像左上角
                # 得到中心点相对于右下角的坐标
                gxi = gain[[2, 3]] - gxy
                """
                把相对于各个网格左上角x<0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
                也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),
                也就是下面的gij = (gxy - offsets).long()操作；
                再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
                也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，
                也采用该部分相邻的两个网格的anchor进行回归；

                也就是说，除了targets本身，还在周围的四个去榆中增加了两个正样本，

                原yolov3就仅仅采用当前网格的anchor进行回归；
                估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，
                是通过对sigmoid输出乘以一个大于1的系数；
                这也与yolov5新的边框回归公式相关；
                由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
                所以中心点回归的公式变为：
                xy.sigmoid() * 2. - 0.5 + cx
                """
                # gxy % 1 意思是求得坐标xy后的小数点，也就是相对于每个网格的偏移量 j 代表x,  k代表y
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # gxi % 1 < g  偏移量不能超过0.5
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j, k, l, m的shape [M,]
                # [5, M] 包括target中心点和它的四个相邻网络（下右上左）
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # [num_targets * 3, 7] 标签数量 * 3
                # t.shape = [5, M, 7], 得到筛选出的框（N, 7）, N 为筛选后的个数
                t = t.repeat((5, 1, 1))[j]
                # 添加偏移量 ： [1, M, 2] + [5, 1, 2] = [5, M, 2] --[j]-->[N, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # b 为batch 中哪一张图片的索引，c为类别
            b, c = t[:, :2].long().T  # image, class
            # 中心点回归标签
            gxy = t[:, 2:4]  # grid xy
            # wh回归标签
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # 添加索引，方便计算损失的时候取出对应位置的输出
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  
            # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # 添加锚框
            anch.append(anchors[a])  # anchors
            # 添加 cls_id
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
