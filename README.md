![FlagAI](assets/imgs/logo.png)
----------
### FlagPerf
[![Lint Code Base](https://github.com/FlagOpen/FlagPerf/actions/workflows/super-linter.yml/badge.svg)](https://github.com/FlagOpen/FlagPerf/actions/workflows/super-linter.yml)

FlagPerf是一款面向AI异构芯片的通用基准测试平台。我们希望探索开源、开放、灵活、公正、客观的AI芯片评测体系，提供行业价值，促进AI产业生态发展。
更多模型及框架支持持续开发中，欢迎加入共同建设，助力AI产业生态发展。

----------
### 支持列表

under review表示对应case的支持已开发完毕，在review中；Incoming表示正在添加或计划添加中；N/A表示不支持或尚无计划添加

#### 训练列表

你可以点击**模型或训练框架**来跳转到**对应case的训练脚本**，✅来跳转到**对应厂商的运行配置**。

<table width="960" border="0" cellpadding="0" cellspacing="0" style='width:960pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="73.60" style='mso-width-source:userset;mso-width-alt:3588;'/>
   <col width="70" style='mso-width-source:userset;mso-width-alt:3413;'/>
   <col width="200.75" style='mso-width-source:userset;mso-width-alt:9788;'/>
   <col width="195.80" style='mso-width-source:userset;mso-width-alt:9547;'/>
   <col width="185.40" style='mso-width-source:userset;mso-width-alt:9040;'/>
   <tr height="16.80" class="xl65" style='height:16.80pt;'>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" x:str>模型</td> 
    <td class="xl65" x:str>框架</td>
    <td class="xl65" x:str>英伟达</td>
    <td class="xl65" x:str>昆仑芯</td>
    <td class="xl65" x:str>天数智芯</td>
    <td class="xl65" x:str>华为昇腾</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="3" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bert" style="text-decoration:none" target="_parent">BERT</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bert/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bert-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bert-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bert-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bert/paddle" style="text-decoration:none" target="_parent">Paddle</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bert-paddle" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bert-paddle" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bert-paddle" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/cpm" style="text-decoration:none" target="_parent">CPM</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/cpm/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/cpm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/cpm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/cpm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="50.40" rowspan="1" style='height:50.40pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/glm" style="text-decoration:none" target="_parent">GLM(medium)</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/glm/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/glm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/glm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/glm-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="50.40" rowspan="3" style='height:50.40pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/resnet50" style="text-decoration:none" target="_parent">ResNet50</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/resnet50/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/resnet50-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/resnet50-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/resnet50-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/resnet50/tensorflow2" style="text-decoration:none" target="_parent">TensorFlow2</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/resnet50-tensorflow2" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>under review</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="2" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mobilenetv2" style="text-decoration:none" target="_parent">MobileNetV2</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mobilenetv2/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/mobilenetv2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/mobilenetv2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/mobilenetv2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr>   
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/vit" style="text-decoration:none" target="_parent">VisionTransformer</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/vit/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/vit-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/vit-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/vit-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="2" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/faster_rcnn" style="text-decoration:none" target="_parent">FasterRCNN</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/faster_rcnn/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/faster_rcnn-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/faster_rcnn-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/faster_rcnn-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
      <td class="xl69" x:str>N/A</td>
   </tr>
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr>    
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bigtransfer" style="text-decoration:none" target="_parent">BigTransfer</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bigtransfer/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bigtransfer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bigtransfer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bigtransfer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
       <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/wav2vec2" style="text-decoration:none" target="_parent">Wav2Vec2</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/wav2vec2/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/wav2vec2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69"><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/wav2vec2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>N/A</td>
   <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/efficientnet" style="text-decoration:none" target="_parent">EfficientNet</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/efficientnet/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/efficientnet-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/efficientnet-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/efficientnet-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
       <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mask_rcnn" style="text-decoration:none" target="_parent">MaskRCNN</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mask_rcnn/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/mask_rcnn-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/mask_rcnn-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
       <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/retinanet" style="text-decoration:none" target="_parent">RetinaNet</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/retinanet/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/retinanet-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/retinanet-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
       <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/tacotron2" style="text-decoration:none" target="_parent">Tacotron2</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/tacotron2/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/tacotron2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/tacotron2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/tacotron2-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
       <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/WaveGlow" style="text-decoration:none" target="_parent">WaveGlow</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/WaveGlow/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/WaveGlow-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69"><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/WaveGlow-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>N/A</td>
       <td class="xl69" x:str>N/A</td>
   </tr>
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/transformer" style="text-decoration:none" target="_parent">Transformer</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/transformer/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/transformer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69"><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/transformer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/transformer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
   <td class="xl69" x:str>N/A</td>
   </tr>
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/swin_transformer" style="text-decoration:none" target="_parent">SwinTransformer</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/swin_transformer/pytorch" style="text-decoration:none" target="_parent">PyTorch</a></td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/swin_transformer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69">N/A</td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/swin_transformer-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
   <td class="xl69" x:str>N/A</td>
   </tr>
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>DLRM</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>RNNT</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="2" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>Llama(7B)</td>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>Paddle</td>
    <td class="xl69" x:str>under review</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr>  
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="2" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>Llama(13B)</td>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl69" x:str>Paddle</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr>  
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>GLM(6B)</td>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>GLM(13B)</td>
    <td class="xl69" x:str>MindSpore</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>Incoming</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>GPT3(6.7B)</td>
    <td class="xl69" x:str>Paddle</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>GPT3(13B)</td>
    <td class="xl69" x:str>Paddle</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>yolov5_large</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>T5_small</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/t5_small-pytorch" style="text-decoration:none" target="_parent">✅</a></td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>GPT2</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>under review</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>TransformerXL</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>whisper</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>DistilBERT</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>Roberta</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>DETR</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
<tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" rowspan="1" style='height:33.60pt;border-right:none;border-bottom:none;' x:str>Longformer</td>
    <td class="xl69" x:str>PyTorch</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
      <td class="xl69" x:str>N/A</a></td>
   </tr> 
  </table>

#### 推理列表

你可以点击**模型来跳转到**对应case的推理脚本及结果。

<table width="960" border="0" cellpadding="0" cellspacing="0" style='width:960pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="73.60" style='mso-width-source:userset;mso-width-alt:3588;'/>
   <col width="70" style='mso-width-source:userset;mso-width-alt:3413;'/>
   <col width="200.75" style='mso-width-source:userset;mso-width-alt:9788;'/>
   <col width="195.80" style='mso-width-source:userset;mso-width-alt:9547;'/>
   <col width="185.40" style='mso-width-source:userset;mso-width-alt:9040;'/>
   <tr height="16.80" class="xl65" style='height:16.80pt;'>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" x:str>模型</td>
    <td class="xl65" x:str>英伟达+tensorrt/inductor</td>
    <td class="xl65" x:str>昆仑芯+xtcl</td>
    <td class="xl65" x:str>天数智芯+ixrt</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/resnet50" style="text-decoration:none" target="_parent">resnet50</a></td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>f32</td>
    <td class="xl69" x:str>f16</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/bertLarge" style="text-decoration:none" target="_parent">BertLarge</a></td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>under review</td>
    <td class="xl69" x:str>Incoming</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/vit_l_16" style="text-decoration:none" target="_parent">VisionTransformer</a></td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>under review</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/yolov5" style="text-decoration:none" target="_parent">Yolov5_large</a></td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>Incoming</td>
   </tr>
   <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/stable_diffusion_v1_4" style="text-decoration:none" target="_parent">Stable Diffusion v1.4</a></td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/swinTransformer" style="text-decoration:none" target="_parent">SwinTransformer</td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/tree/main/inference/benchmarks/llama2_7b_mmlu" style="text-decoration:none" target="_parent">Llama2-7B-mmlu</td>
    <td class="xl69" x:str>f32/f16</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60" style='height:33.60pt;border-right:none;border-bottom:none;' x:str><a href="https://github.com/FlagOpen/FlagPerf/pull/209" style="text-decoration:none" target="_parent">Aquila-7B-mmlu</td>
    <td class="xl69" x:str>fp16</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
    <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str>DLRM</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
        <tr height="16.80" style='height:16.80pt;'>
    <td class="xl65" height="33.60"  style='height:33.60pt;border-right:none;border-bottom:none;' x:str>RNNT</td>
    <td class="xl69" x:str>Incoming</td>
    <td class="xl69" x:str>N/A</td>
    <td class="xl69" x:str>N/A</td>
   </tr>
</table>


### 训练部署及启动说明

[训练文档](https://github.com/FlagOpen/FlagPerf/tree/main/training/README.md)

### 推理部署及启动说明

[推理文档](https://github.com/FlagOpen/FlagPerf/blob/main/docs/dev/inference-case-doc.md)

### 贡献代码

本项目目前由北京智源人工智能研究院、天数智芯、百度PaddlePaddle、昆仑芯、华为昇腾、华为昇思MindSpore共同建设中。
诚邀各框架、芯片、编译器团队与个人参与！

![cooperation](assets/imgs/logos.png)

### 联系我们

flagperf@baai.ac.cn
### 许可证
本项目基于Apache 2.0 license。 
<br>本项目的代码来源于不同的代码仓库，关于各模型测试Case的情况，请参考各模型测试Case目录的文档。
