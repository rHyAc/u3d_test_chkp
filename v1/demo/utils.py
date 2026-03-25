import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)



class cap_bdg(nn.Module):
    def __init__(self, c, n, cout, acti):
        super(cap_bdg, self).__init__()

        self.n = n
        self.w = nn.Conv2d(c, c*n, 1)

        self.mask_block = nn.Sequential(
            nn.GroupNorm(c//4, c),
            nn.ReLU(True),
            nn.Conv2d(c, 64, 3, padding=1),
            nn.GroupNorm(64//4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1)
        )

        self.gen_block = nn.Sequential(
            nn.InstanceNorm2d(n, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n, n, 1),
            nn.InstanceNorm2d(n, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n, cout, 1)
        )
    
        self.acti = acti()

    def forward(self, sr, tg):
        b, c, h, w = sr.shape
        sr = self.w(sr).view(b, -1, c, h*w)
        op = sr.sum(3, True)
        for _ in range(3):
            cof = F.softmax((sr * F.normalize(op, dim=2)).sum(2, True), 3)
            op = (sr * cof).sum(3, True)
        
        msk_op = op.unsqueeze(4)
        msk_tg = tg.unsqueeze(1)
        msk_tg = msk_tg + msk_op
        msk_tg = msk_tg.view(-1, c, h, w)
        msk = self.mask_block(msk_tg).view(b, -1, h, w)

        hw_cof = F.softmax(msk.view(b, self.n, -1), -1).view(b, self.n, h, w)[..., None]
        hw_idx = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h).to('cuda:0'), torch.linspace(-1, 1, w).to('cuda:0')), -1)[None, None, ...].repeat(b, self.n, 1, 1, 1)
        hw_centr = (hw_cof * hw_idx).sum((2, 3), True)
        centr_loss = (hw_cof * (hw_idx - hw_centr) ** 2).sum((2, 3, 4)).mean()

        rec = self.acti(self.gen_block(msk))
        op = torch.einsum('ijkz, ijmn -> ijkmn', op, msk)
        return op, msk, rec, centr_loss, cof.view(b, -1, h, w)




class EDDeconv(nn.Module):
    def __init__(self, cin, cout, zdim=512, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        
        self.e_c0 = nn.Conv2d(cin, nf, 3, padding=1)
        self.e_gn0 = nn.GroupNorm(nf//4, nf)
        self.e_rl0 = nn.ReLU(True)
        self.e_d0 = nn.AvgPool2d(2)

        self.e_c1 = nn.Conv2d(nf, nf*2, 3, padding=1)
        self.e_gn1 = nn.GroupNorm(nf*2//4, nf*2)
        self.e_rl1 = nn.ReLU(True)
        self.e_d1 = nn.AvgPool2d(2)

        self.e_c2 = nn.Conv2d(nf*2, nf*4, 3, padding=1)
        self.e_gn2 = nn.GroupNorm(nf*4//4, nf*4)
        self.e_rl2 = nn.ReLU(True)
        self.e_d2 = nn.AvgPool2d(2)

        self.e_c3 = nn.Conv2d(nf*4, nf*8, 3, padding=1)
        self.e_gn3 = nn.GroupNorm(nf*8//4, nf*8)
        self.e_rl3 = nn.ReLU(True)
        self.e_d3 = nn.AvgPool2d(2)
        
        self.e_l = nn.Conv2d(nf*8, zdim, kernel_size=4)


        self.d_l = nn.ConvTranspose2d(zdim, nf*8, kernel_size=4)
        self.d_rlz = nn.ReLU(True)
        self.d_cy = nn.Conv2d(nf*8, nf*8, 3, padding=1)
        self.d_gny = nn.GroupNorm(nf*8//4, nf*8)
        self.d_rly = nn.ReLU(True)

        self.d_u0 = nn.Upsample(scale_factor=2)
        self.d_dc0 = nn.Conv2d(nf*8, nf*8, 3, padding=1, groups=nf*8)
        self.d_dc_gn0 = nn.GroupNorm(nf*8//4, nf*8)
        self.d_dc_rl0 = nn.ReLU(True)
        self.d_pc0 = nn.Conv2d(nf*8, nf*4, 1)
        self.d_pc_gn0 = nn.GroupNorm(nf*4//4, nf*4)
        
        self.d_u1 = nn.Upsample(scale_factor=2)
        self.d_dc1 = nn.Conv2d(nf*4, nf*4, 3, padding=1, groups=nf*4)
        self.d_dc_gn1 = nn.GroupNorm(nf*4//4, nf*4)
        self.d_dc_rl1 = nn.ReLU(True)
        self.d_pc1 = nn.Conv2d(nf*4, nf*2, 1)
        self.d_pc_gn1 = nn.GroupNorm(nf*2//4, nf*2)

        self.d_u2 = nn.Upsample(scale_factor=2)
        self.d_dc2 = nn.Conv2d(nf*2, nf*2, 3, padding=1, groups=nf*2)
        self.d_dc_gn2 = nn.GroupNorm(nf*2//4, nf*2)
        self.d_dc_rl2 = nn.ReLU(True)
        self.d_pc2 = nn.Conv2d(nf*2, nf, 1)
        self.d_pc_gn2 = nn.GroupNorm(nf//4, nf)

        self.d_u3 = nn.Upsample(scale_factor=2)
        self.d_c3 = nn.Conv2d(nf, nf, 3, padding=1)
        self.d_gn3 = nn.GroupNorm(nf//4, nf)
        self.d_rl3 = nn.ReLU(True)

        self.d_c4 = nn.Conv2d(nf, nf, 5, padding=2)
        self.d_gn4 = nn.GroupNorm(nf//4, nf)
        self.d_rl4 = nn.ReLU(True)

        self.d_o = nn.Conv2d(nf, cout, 5, padding=2)
        
        if activation is not None:
            im_acti = activation
            self.acti = activation()
        else:
            im_acti = nn.Identity
            self.acti = nn.Identity()

        self.cb_0 = cap_bdg(nf*8, 7, cout, im_acti)
        self.cb_1 = cap_bdg(nf*4, 7, cout, im_acti)
        self.cb_2 = cap_bdg(nf*2, 7, cout, im_acti)
        self.cb_3 = cap_bdg(nf, 7, cout, im_acti)

        self.grad_list = []
    
    def grad_h(self, grad):
        self.grad_list.append(grad.abs().mean())

    def forward(self, input):   
        x0 = self.e_c0(input)               # (64, 64, 64)
        if x0.requires_grad:
            x0.register_hook(self.grad_h)
        x = self.e_gn0(x0)
        x = self.e_rl0(x)
        x = self.e_d0(x)

        x1 = self.e_c1(x)                   # (128, 32, 32)
        if x1.requires_grad:
            x1.register_hook(self.grad_h)
        x = self.e_gn1(x1)
        x = self.e_rl1(x)
        x = self.e_d1(x)

        x2 = self.e_c2(x)                   # (256, 16, 16)
        if x2.requires_grad:
            x2.register_hook(self.grad_h)
        x = self.e_gn2(x2)
        x = self.e_rl2(x)
        x = self.e_d2(x)

        x3 = self.e_c3(x)                   # (512, 8, 8)
        if x3.requires_grad:
            x3.register_hook(self.grad_h)
        x = self.e_gn3(x3)
        x = self.e_rl3(x)
        x = self.e_d3(x)

        x = self.e_l(x)


        x = self.d_l(x)
        x = self.d_rlz(x)                        
        x = self.d_cy(x)
        x = self.d_gny(x)
        x = self.d_rly(x)

        x = self.d_u0(x)
        xd = self.d_dc0(x)                   # (512, 8, 8) 
        x3, msk3, rec3, cl3, cf3 = self.cb_0(x3, xd)
        x = self.d_dc_gn0(xd + x3.sum(1))
        x = self.d_dc_rl0(x)
        x = self.d_pc0(x)
        x = self.d_pc_gn0(x)                  

        x = self.d_u1(x)
        xc = self.d_dc1(x)                   # (256, 16, 16)
        x2, msk2, rec2, cl2, cf2 = self.cb_1(x2, xc)
        x = self.d_dc_gn1(xc + x2.sum(1))
        x = self.d_dc_rl1(x)
        x = self.d_pc1(x)
        x = self.d_pc_gn1(x)    

        x = self.d_u2(x)
        xb = self.d_dc2(x)                   # (128, 32, 32)
        x1, msk1, rec1, cl1, cf1 = self.cb_2(x1, xb)
        x = self.d_dc_gn2(xb + x1.sum(1))
        x = self.d_dc_rl2(x)
        x = self.d_pc2(x)
        x = self.d_pc_gn2(x)    

        x = self.d_u3(x)    
        xa = self.d_c3(x)                    # (64, 64, 64)
        x0, msk0, rec0, cl0, cf0 = self.cb_3(x0, xa)
        x = self.d_gn3(xa + x0.sum(1))
        x = self.d_rl3(x)

        x = self.d_c4(x)
        x = self.d_gn4(x)
        x = self.d_rl4(x)

        x = self.d_o(x)

        return self.acti(x), [msk3, msk2, msk1, msk0], [(xd, x3), (xc, x2), (xb, x1), (xa, x0)], [rec3, rec2, rec1, rec0], [cl3, cl2, cl1, cl0], [cf3, cf2, cf1, cf0]



class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)




IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def save_video(out_fold, frames, fname='image', ext='.mp4', cycle=False):
    os.makedirs(out_fold, exist_ok=True)
    frames = frames.detach().cpu().numpy().transpose(0,2,3,1)  # TxCxHxW -> TxHxWxC
    if cycle:
        frames = np.concatenate([frames, frames[::-1]], 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vid = cv2.VideoWriter(os.path.join(out_fold, fname+ext), fourcc, 25, (frames.shape[2], frames.shape[1]))
    [vid.write(np.uint8(f[...,::-1]*255.)) for f in frames]
    vid.release()


def save_image(out_fold, img, fname='image', ext='.png', msk=False, map=False):
    os.makedirs(out_fold, exist_ok=True)

    if not msk:
        img = img.detach().cpu().numpy().transpose(1,2,0)
        if 'depth' in fname:
            im_out = np.uint16(img*65535.)
        else:
            im_out = np.uint8(img*255.)
        cv2.imwrite(os.path.join(out_fold, fname+ext), im_out[:,:,::-1])
    
    else:
        tg_dir = os.path.join(out_fold, fname)
        os.makedirs(tg_dir, exist_ok=True)
        for i, mk in enumerate(img):
            mk = mk[0]
            for j, mk_img in enumerate(mk):
                if not map:
                    im_out = np.uint8(mk_img * 255)
                    cv2.imwrite(os.path.join(tg_dir, 'L' + str(i) + '_' + str(j) + ext), im_out)
                else:
                    np.save(os.path.join(tg_dir, 'L' + str(i) + '_' + str(j)), mk_img)



def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def export_to_obj_string(vertices, normal):
    b, h, w, _ = vertices.shape
    vertices[:,:,:,1:2] = -1*vertices[:,:,:,1:2]  # flip y
    vertices[:,:,:,2:3] = 1-vertices[:,:,:,2:3]  # flip and shift z
    vertices *= 100
    vertices_center = nn.functional.avg_pool2d(vertices.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertices = torch.cat([vertices.view(b,h*w,3), vertices_center.view(b,(h-1)*(w-1),3)], 1)

    vertice_textures = get_grid(b, h, w, normalize=True)  # BxHxWx2
    vertice_textures[:,:,:,1:2] = -1*vertice_textures[:,:,:,1:2]  # flip y
    vertice_textures_center = nn.functional.avg_pool2d(vertice_textures.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_textures = torch.cat([vertice_textures.view(b,h*w,2), vertice_textures_center.view(b,(h-1)*(w-1),2)], 1) /2+0.5  # Bx(H*W)x2, [0,1]

    vertice_normals = normal.clone()
    vertice_normals[:,:,:,0:1] = -1*vertice_normals[:,:,:,0:1]
    vertice_normals_center = nn.functional.avg_pool2d(vertice_normals.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_normals_center = vertice_normals_center / (vertice_normals_center**2).sum(3, keepdim=True)**0.5
    vertice_normals = torch.cat([vertice_normals.view(b,h*w,3), vertice_normals_center.view(b,(h-1)*(w-1),3)], 1)  # Bx(H*W)x2, [0,1]

    idx_map = torch.arange(h*w).reshape(h,w)
    idx_map_center = torch.arange((h-1)*(w-1)).reshape(h-1,w-1)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces2 = torch.stack([idx_map[1:,:w-1], idx_map[1:,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces3 = torch.stack([idx_map[1:,1:], idx_map[:h-1,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces4 = torch.stack([idx_map[:h-1,1:], idx_map[:h-1,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces = torch.cat([faces1, faces2, faces3, faces4], 1)

    objs = []
    mtls = []
    for bi in range(b):
        obj = "# OBJ File:"
        obj += "\n\nmtllib $MTLFILE"
        obj += "\n\n# vertices:"
        for v in vertices[bi]:
            obj += "\nv " + " ".join(["%.4f"%x for x in v])
        obj += "\n\n# vertice textures:"
        for vt in vertice_textures[bi]:
            obj += "\nvt " + " ".join(["%.4f"%x for x in vt])
        obj += "\n\n# vertice normals:"
        for vn in vertice_normals[bi]:
            obj += "\nvn " + " ".join(["%.4f"%x for x in vn])
        obj += "\n\n# faces:"
        obj += "\n\nusemtl tex"
        for f in faces[bi]:
            obj += "\nf " + " ".join(["%d/%d/%d"%(x+1,x+1,x+1) for x in f])
        objs += [obj]

        mtl = "newmtl tex"
        mtl += "\nKa 1.0000 1.0000 1.0000"
        mtl += "\nKd 1.0000 1.0000 1.0000"
        mtl += "\nKs 0.0000 0.0000 0.0000"
        mtl += "\nd 1.0"
        mtl += "\nillum 0"
        mtl += "\nmap_Kd $TXTFILE"
        mtls += [mtl]
    return objs, mtls
