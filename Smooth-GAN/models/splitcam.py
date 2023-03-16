import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.networks.utils import batch_conv2d, batch_transposeconv2d
import pdb


def hardmax(similar):
    val_max, id_max = torch.max(similar, 1)
    num = similar.size(1)
    sb = torch.Tensor(range(num)).long().to(similar.device)
    id_max = id_max[:, None, :, :]
    sb = sb[None, ..., None, None]
    similar = (sb==id_max).float().detach()
    return similar

class ReduceContextAttentionP1(nn.Module):
    def __init__(self, bkg_patch_size=4, stride=1, ufstride=1, 
            softmax_scale=10., nn_hard=False, pd=1,
                 fuse_k=3, is_fuse=False,
                 th=0.5, norm_type=1, is_th=False):
        super(ReduceContextAttentionP1, self).__init__()
        self.bkg_patch_size = bkg_patch_size
        self.nn_hard = nn_hard
        self.stride = stride
        self.ufstride = ufstride
        self.softmax_scale = softmax_scale
        self.forward = self.forward_batch
        self.pd = pd
        self.fuse_k = fuse_k
        self.is_fuse = is_fuse
        self.th = th
        self.is_th = is_th
        self.norm_type = norm_type
        #self.forward = self.forward_test

    def get_conv_kernel(self, x, mask=None):
        batch, c, h_small, w_small = x.shape
        if self.norm_type == 1:
            x = x / torch.sqrt((x**2).sum(3, keepdim=True).sum(2, keepdim=True) + 1e-8)
        _x = F.pad(x, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        kernel = F.unfold(input=_x, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        kernel = kernel.transpose(1, 2) \
            .view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        if self.norm_type == 2:
            kernel = kernel/ torch.sqrt(
                    (kernel**2).sum(3, keepdim=True).sum(4, keepdim=True)+1e-8)
        # b*hw*c*k*c
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.squeeze(2)
        mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)).float()
        #mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)==1).float()
        return kernel, mm

    def forward_batch(self, f, b, mask=None):
        batch, c, h, w = b.shape
        batch, c, h_small, w_small = f.shape
        if mask is None:
            mask = torch.ones(batch, 1, h_small, w_small).to(f.device)
        else:
            mask = 1-mask
        # mask valid region
        softmax_scale = self.softmax_scale
        kernel, mmk = self.get_conv_kernel(b, mask)
        # mmk: valid ratio of each bkg patch
        _f = F.pad(f, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        cos_similar = batch_conv2d(_f, weight=kernel, stride=self.stride)
        _, cs, hs, ws = cos_similar.shape
        hb, wb = h//2, w//2

        if self.is_fuse:
            fuse_weight = torch.eye(self.fuse_k).to(f.device)
            fuse_weight = fuse_weight[None, None, ...]
            cos_similar = cos_similar.view(-1, cs, hs*ws)[:, None, ...]
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.reshape(batch, 1, cs, hs*ws)
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.squeeze(1)
            cos_similar = cos_similar.reshape(batch, cs, hs, ws)

        if self.is_th:
            mm = (mmk>self.th).float()
        else:
            _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
            m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), \
                         stride=self.stride)
            m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
            m = m.squeeze(2)
            mmp = (m.mean(3).mean(2)).float()
            mmp = mmp.view(batch, 1, hs, ws) # mmp: valid ratio of fg patch
            mm = (mmk>mmp).float()  # replace with more valid
            ppp = (mmp>self.th).float() # ppp: mask of partial valid
            mm = mm*ppp # partial valid being replaced with more valid
            mm = mm + (mmk==1).float().expand_as(mm)  # and full valid
            mm = (mm>0).float()
        cos_similar = cos_similar * mm
        cos_similar = F.softmax(cos_similar*softmax_scale, dim=1)
        if self.nn_hard:
            cos_similar = hardmax(cos_similar)
        return cos_similar

class ReduceContextAttentionP2(nn.Module):
    def __init__(self, bkg_patch_size=16, stride=8, ufstride=8, pd=4, mk=True):
        super(ReduceContextAttentionP2, self).__init__()
        self.stride = stride
        self.bkg_patch_size = bkg_patch_size
        self.forward = self.forward_batch
        self.ufstride = ufstride
        self.pd = pd
        self.mk = mk
        #self.forward = self.forward_test
        self.stride_aux = stride
        self.aux_patch_size = bkg_patch_size
        self.ufstride_aux = ufstride

    def get_aux_kernel(self, b):
        batch, c, h, w = b.shape
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.aux_patch_size, self.aux_patch_size),
                              stride=self.ufstride_aux)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(batch, -1, c, self.aux_patch_size, self.aux_patch_size)
        return bkg_kernel

    def get_deconv_kernel(self, b, mask):
        batch, c, h, w = b.shape
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        msk_kernel = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        msk_kernel = msk_kernel.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        if self.mk:
            bkg_kernel = bkg_kernel*(1-msk_kernel)

        return bkg_kernel, msk_kernel

    def forward_batch(self, cos_similar, b, mask, dict_aux):
        # use original background for reconstruction
        _, _, hs, ws = cos_similar.shape
        bkg_kernel, msk_kernel = self.get_deconv_kernel(b, mask)
        #hard_similar = hardmax(cos_similar.detach())
        output = batch_transposeconv2d(cos_similar,
                                       weight=bkg_kernel,stride=self.stride)

        norm_kernel = torch.ones(1, 1, self.bkg_patch_size, self.bkg_patch_size).to(mask.device)
        weight_map = torch.ones(1, 1, hs, ws).to(mask.device)
        weight_map = F.conv_transpose2d(weight_map, norm_kernel, stride=self.stride)
        mask_recon = batch_transposeconv2d(cos_similar,
                                           weight=msk_kernel,stride=self.stride)
        mask_recon = mask_recon / weight_map
        if self.pd > 0:
            output = output[:,:,self.pd:-self.pd,self.pd:-self.pd]
            mask_recon = mask_recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
        recon_aux = {"hole":mask_recon}
        for k,v in dict_aux.items():
            hard_similar = hardmax(cos_similar)
            kernel = self.get_aux_kernel(v)
            recon = batch_transposeconv2d(hard_similar,
                                          weight=kernel,stride=self.stride_aux)
            recon = recon / weight_map
            if self.pd > 0:
                recon = recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
            recon_aux[k] = recon
        return output,recon_aux

class AttUpLayer(nn.ModuleDict):
    def __init__(self, patch_size=4, grid_size=32, 
                 th=0.5, norm_type=1, is_th=False,
                 mk=True
            ):
        super().__init__()
        self.cam_1 = ReduceContextAttentionP1(nn_hard=False,
                ufstride=patch_size,
                stride=patch_size,
                bkg_patch_size=patch_size, pd=0,
                norm_type=norm_type,
                is_th=is_th, th=th
                )
        self.cam_2 = ReduceContextAttentionP2(
                ufstride=patch_size,
                bkg_patch_size=patch_size,
                stride=patch_size, pd=0,mk=mk)
        scale_rate=2
        #self.cam_2_v = ReduceContextAttentionP2(
        #        ufstride=patch_size*scale_rate,
        #        bkg_patch_size=patch_size*scale_rate,
        #        stride=patch_size*scale_rate, pd=0,mk=False)
        self.sample_factor = patch_size*scale_rate
        self.patch_size = patch_size
        self.grid_size = grid_size

    def unfold(self, input):
        b = input.size(0)
        output = F.unfold(input=input, kernel_size=(self.grid_size,self.grid_size), stride=self.grid_size)
        n_grid = output.size(2)
        output = output.transpose(1, 2).reshape(b*n_grid, -1, self.grid_size, self.grid_size)
        return output

    def forward(self, q_im, k_im, v_im, up_im, msk_q, msk_up):
        b,c,hs,ws = q_im.shape
        _,_,ht,wt = up_im.shape
        assert ht%hs==0 and wt%ws==0
        assert ht//hs == wt//ws
        assert hs%self.patch_size==0
        assert ws%self.patch_size==0
        assert hs%self.grid_size==0
        assert ws%self.grid_size==0

        sf = ht//hs*self.patch_size

        if min(hs,ws)>self.grid_size:
            n_h = hs//self.grid_size
            n_w = ws//self.grid_size
            _q = self.unfold(q_im)
            _k = self.unfold(k_im)
            _v = self.unfold(v_im)
            _m = self.unfold(msk_q)
            _similar = self.cam_1(_q, _k, _m)
            _,_,s_h, s_w = _similar.shape
            _similar=_similar.view(b,n_h*n_w,s_h,s_w,s_h,s_w)
            sb = _similar.view(b,n_h*n_w,-1).transpose(1,2).reshape(b,s_h,s_w,s_h,s_w,n_h*n_w)
            ss = torch.diag_embed(sb, 0, 1, 4)
            #ss = torch.zeros(1,4,8,8,4,8,8).to(similar.device)
            #ss[0,[0,1,2,3],:,:,[0,1,2,3],:,:] = _similar
            ss = ss.reshape(b,n_h,n_w,s_h,s_w,n_h*n_w,s_h,s_w).transpose(2,3).reshape(b,n_h*s_h,n_w*s_w,n_h*n_w,s_h,s_w)
            ss = ss.reshape(b,n_h*s_h,n_w*s_w,n_h,n_w,s_h,s_w).transpose(4,5).reshape(b,n_h*s_h,n_w*s_w,n_h*s_h,n_w*s_w)
            similar=ss.view(b,n_h*s_h*n_w*s_w,n_h*s_h,n_w*s_w)
        else:
            similar = self.cam_1(q_im, k_im, msk_q)
        recon, recon_aux = self.cam_2(similar, v_im, msk_q, {})
        #recon, recon_aux = self.cam_2_v(similar, up_im, msk_up, {})

        label = similar.argmax(1)[:,None,...].detach()
        label_up = F.interpolate(label.float(), scale_factor=sf, mode='nearest')[:,0]
        offset_y,offset_x = torch.meshgrid(torch.arange(ht),torch.arange(wt))
        #_, recon_aux = self.cam_2_v(similar, up_im, msk_up, 
        #        {'xy':torch.stack((offset_x, offset_y), 0)[None].float().cuda()})
        x =label_up%(wt//sf)*sf+offset_x.to(label.device)%sf
        y =label_up//(wt//sf)*sf+offset_y.to(label.device)%sf
        x = (x+1)/float(wt+1)*2-1
        y = (y+1)/float(ht+1)*2-1
        recon_grid = F.grid_sample(up_im, torch.stack((x,y),3), mode='nearest')

        return recon, recon_grid


def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Define batch convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, out_channel, in_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


def batch_transposeconv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Define batch transposed convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, in_channel, out_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, in_channels, out_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(in_channels*b_i, out_channels, kernel_height_size, kernel_width_size)

    out = F.conv_transpose2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding, output_padding=output_padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out