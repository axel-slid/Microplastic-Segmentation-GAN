# Microplastic-Segmentation-GAN

Current methods for microplastic identification in water samples are costly and
require expert analysis. Here, we propose a deep learning segmentation model
to automatically identify microplastics in microscopic images. We obtain images
from the Moore Institute for Plastic Pollution Research and employ a Generative
Adversarial Network (GAN) to supplement and generate diverse training data. To
verify the validity of the generated data, we conducted a reader study where an
expert was able to discern the generated microplastic from real microplastic at a
rate of 68%. Our segmentation model trained on the combined data achieved an
F1-Score of 0.91 on a diverse dataset, compared to the model without generated
data’s 0.81. With our findings we aim to enhance the ability of both experts and
citizens to detect microplastic in water across diverse ecological contexts, thereby
improving the cost and accessibility of microplastic analysis
