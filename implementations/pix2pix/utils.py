import torch,rawpy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def apply_wb(org_img,pred,pred_type):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """
    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] * (1 / (pred[:,0,:,:]+1e-8))    # R_wb = R * (1/illum_R)
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] * (1 / (pred[:,2,:,:]+1e-8))    # B_wb = B * (1/illum_B)
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb

def rgb2uvl(img_rgb):
    """
    convert 3 channel rgb image into uvl
    """
    epsilon = 1e-8
    img_uvl = np.zeros_like(img_rgb, dtype='float32')
    img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
    img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
    img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

    return img_uvl

def plot_illum(pred_map=None,gt_map=None,MAE_illum=None,MAE_rgb=None,PSNR=None):
    """
    plot illumination map into R,G 2-D space
    """

    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'ro')
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'bx')
    plt.title(f"MAE: {MAE_illum:.5f} / PSNR: {PSNR:.3f}")
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def mix_chroma(mixmap,chroma_list,illum_count):
    """
    Mix illuminant chroma according to mixture map coefficient
    mixmap      : (w,h,c) - c is the number of valid illuminant
    chroma_list : (3 (RGB), 3 (Illum_idx))
                  contains R,G,B value or 0,0,0
    illum_count : contains valid illuminant number (1,2,3)
    """
    ret = np.stack((np.zeros_like(mixmap[:,:,0]),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret


def visualize(input_patch, pred_patch, gt_patch, templete, concat=True):
    """
    Visualize model inference result.
    1. Re-bayerize RGB image by duplicating G pixels.
    2. Copy bayer pattern image into rawpy templete instance
    3. Use user_wb to render RGB image
    4. Crop proper size of patch from rendered RGB image
    """
    input_patch = input_patch.permute((1,2,0))
    pred_patch = pred_patch.permute((1,2,0))
    gt_patch = gt_patch.permute((1,2,0))

    height, width, _ = input_patch.shape
    raw = rawpy.imread(templete + ".dng")

    white_level = raw.white_level

    if templete == 'sony':
        black_level = 512
        white_level = raw.white_level / 4
    else:
        black_level = min(raw.black_level_per_channel)
        white_level = raw.white_level
        
    input_rgb = input_patch.numpy().astype('uint16')
    output_rgb = np.clip(pred_patch.cpu().numpy(), 0, white_level).astype('uint16')
    gt_rgb = gt_patch.numpy().astype('uint16')

    input_bayer = bayerize(input_rgb, templete, black_level)
    output_bayer = bayerize(output_rgb, templete, black_level)
    gt_bayer = bayerize(gt_rgb, templete, black_level)

    input_rendered = render(raw, white_level, input_bayer, height, width, "daylight_wb")
    output_rendered = render(raw, white_level, output_bayer, height, width, "maintain")
    gt_rendered = render(raw, white_level, gt_bayer, height, width, "maintain")

    if concat:
        return np.hstack([input_rendered, output_rendered, gt_rendered])
    else:
        return input_rendered, output_rendered, gt_rendered

def bayerize(img_rgb, camera, black_level):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))
    
    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B
    elif camera == "sony" or camera == 'nikon':
        bayer_pattern[0::2,0::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,2] # B

    return bayer_pattern + black_level

def render(raw, white_level, bayer, height, width, wb_method):
    raw_mat = raw.raw_image
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    if wb_method == "maintain":
        user_wb = [1.,1.,1.,1.]
    elif wb_method == "daylight_wb":
        user_wb = raw.daylight_whitebalance

    rgb = raw.postprocess(user_sat=white_level, user_wb=user_wb, half_size=True, no_auto_bright=False)
    rgb_croped = rgb[0:height,0:width,:]
    
    return rgb_croped
    
def get_abstract_illum_map(img, pool_size=1):
    pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
    return pool(img)
