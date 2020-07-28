import numpy as np
from matplotlib import pyplot as plt
import os
from torch import nn
from torchvision import transforms
import torchvision.models as models
import torch
# from kornia import color

from kornia_lab import RgbToLab, LabToRgb

from torch.autograd import Variable

import relighters.DPR.preparation as prep
# from faces import face_utils
from relighters.DPR.model.defineHourglass_512_gray_skip import HourglassNet

r2l = RgbToLab()
l2r = LabToRgb()

sh1 = [
    1.084125496282453138e+00,
    -4.642676300617166185e-01,
    2.837846795150648915e-02,
    6.765292733937575687e-01,
    -3.594067725393816914e-01,
    4.790996460111427574e-02,
    -2.280054643781863066e-01,
    -8.125983081159608712e-02,
    2.881082012687687932e-01]

sh2 = [1.084125496282453138e+00,
       -4.642676300617170626e-01,
       5.466255701105990905e-01,
       3.996219229512094628e-01,
       -2.615439760463462715e-01,
       -2.511241554473071513e-01,
       6.495694866016435420e-02,
       3.510322039081858470e-01,
       1.189662732386344152e-01
       ]

sh3 = [1.084125496282453138e+00,
       -4.642676300617179508e-01,
       6.532524688468428486e-01,
       -1.782088862752457814e-01,
       3.326676893441832261e-02,
       -3.610566644446819295e-01,
       3.647561777790956361e-01,
       -7.496419691318900735e-02,
       -5.412289239602386531e-02
       ]

shs = [np.array(sh1) * 0.7, np.array(sh2) * 0.7, np.array(sh3) * 0.7]

path = '../data/'

img_names = ['portrait_a', 'portrait_p', 'portrait_j']

imgs = []
for i, img in enumerate(img_names):
    img_np = prep.load_np_img(img + ".png")
    img_luv = prep.np_rgb_to_torch_lab(img_np.transpose((2, 0, 1)) / 255)

    inputL = img_luv[0, :, :]
    inputL = (inputL / 100.0)
    inputL = inputL[None, None, ...]

    relighting_input = inputL.cuda().float()

    modelFolder = '../relighters/DPR/trained_model/'

    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
    my_network.cuda()
    my_network.train(False)

    sh = shs[i]
    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, outputSH = my_network(relighting_input, sh, 0)
    outputImg = outputImg[0].data.cpu()
    outputImg = outputImg.permute((1, 2, 0))
    outputImg = torch.squeeze(outputImg)
    outputImg = (outputImg * 100.0)

    output_luv = img_luv
    output_luv[0, :, :] = outputImg

    output_rgb = l2r.lab_to_rgb(output_luv)

    # 3, 512, 512 -> 512, 512, 3
    output_rgb = output_rgb.permute(1, 2, 0)

    # Mak sure we don't exceed
    output_rgb = output_rgb.clamp(0., 1.)

    plt.figure()
    plt.imshow(output_rgb)
    plt.axis('off')
    plt.show()
