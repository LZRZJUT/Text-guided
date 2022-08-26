GenerateMask.py 用来生成每一个Entity对应的mask, 输入的位置为./image, 输出的位置为./mask_output, 目前仅支持一次传入一张图片，原因在于生成mask时第二张图片会把第一张图片生成的mask覆盖，目前也不需要一次传入多张照片。
GenerateEntity.py  在原图上生成每一个单独entity的图片，阈值设置为0.4，因为很多0.3、0.2左右的score比较杂乱无章，输入为image和mask,输出位置为./Entity_Results
接下的任务是将entity图片送入CLIP，获得它们和text的相关度指标，根据指标选择对应的mask, 结果是要将所有的mask合成为一张图片，不需要考虑维度。