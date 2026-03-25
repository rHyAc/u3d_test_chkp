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


class EncDown(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.GroupNorm(idim//4, idim),
            nn.Conv2d(idim, odim, 2, 2)
        )

    def forward(self, x):
        return self.layer(x)


class EncBlock(nn.Module):
    def __init__(self, dim, sec=64):
        super().__init__()

        self.lat_dim = 2 * dim

        self.gn = nn.GroupNorm(dim//4, dim)
        self.c1 = nn.Conv2d(dim, self.lat_dim, 1)
        self.act = nn.GELU()
        self.ac = nn.Conv2d(self.lat_dim, self.lat_dim, 3, padding=1, groups=self.lat_dim, bias=False)
        self.c2= nn.Conv2d(self.lat_dim, dim, 1)


    def forward(self, x):
        sc = x

        x = self.gn(x)
        x = self.c1(x)

        x = self.act(x)

        x = self.ac(x)
        x = self.c2(x)
        
        x = x + 2 * sc

        return x
    

class DecBlock(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.blk = nn.Sequential(
            nn.GroupNorm(idim//4, idim),
            nn.Conv2d(idim, odim, 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.blk(x)

        return x
    

class DANet(nn.Module):
    def __init__(self, idim=3, odim=1, lat_dim=[64, 128, 256, 512], act=nn.Tanh):
        super().__init__()

        self.enc_down = nn.ModuleList()
        self.enc_down.append(nn.Conv2d(idim, lat_dim[0], 2, 2))
        for i in range(3):
            self.enc_down.append(EncDown(lat_dim[i], lat_dim[i+1]))
        
        self.enc_blk = nn.ModuleList()
        for i in lat_dim:
            self.enc_blk.append(EncBlock(i))

        self.enc_prj = nn.Conv2d(lat_dim[-1], lat_dim[-1], 4)

        self.dec_init = nn.Sequential(
            nn.GroupNorm(lat_dim[-1]//4, lat_dim[-1]),
            nn.ConvTranspose2d(lat_dim[-1], lat_dim[-1], 4),
            nn.ReLU(True),
            nn.GroupNorm(lat_dim[-1]//4, lat_dim[-1]),
            nn.Conv2d(lat_dim[-1], lat_dim[-1], 3, padding=1),
            nn.ReLU(True)
        )


        self.dec_blk = nn.ModuleList()
        for i in range(3, 0, -1):
            self.dec_blk.append(DecBlock(lat_dim[i], lat_dim[i-1]))
        self.dec_blk.append(DecBlock(lat_dim[0], lat_dim[0]))

        self.dec_head = nn.Sequential(
            nn.GroupNorm(lat_dim[0]//4, lat_dim[0]),
            nn.Conv2d(lat_dim[0], lat_dim[0], 5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(lat_dim[0], lat_dim[0], 5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(lat_dim[0], odim, 1)
        )

        act = act or nn.Identity
        self.act = act()


    def forward(self, x):

        for i in range(2):
            x = self.enc_down[i](x)
            x = self.enc_blk[i](x)
        x = self.enc_down[2](x)
        

        x = self.enc_blk[2](x)
        x = self.enc_down[3](x)
        x = self.enc_blk[3](x)

        x = self.enc_prj(x)

        x = self.dec_init(x)


        for i in range(4):
            x = F.interpolate(x, scale_factor=2)
            x = self.dec_blk[i](x)

        r = self.dec_head(x)
        r1 = self.act(r)

        return r1



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


def save_image(out_fold, img, fname='image', ext='.png'):
    os.makedirs(out_fold, exist_ok=True)
    img = img.detach().cpu().numpy().transpose(1,2,0)
    if 'depth' in fname:
        im_out = np.uint16(img*65535.)
    else:
        im_out = np.uint8(img*255.)
    cv2.imwrite(os.path.join(out_fold, fname+ext), im_out[:,:,::-1])


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
