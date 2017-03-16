# GanadaProject

* 가,나,다~ 하까지의 한 음절 음성을 인식하는 학습기입니다.
  * 학습 Feature로서 MFCC를 사용했습니다.
  * CNN를 구성했습니다. (2중 ConvLayer + 2중 DenseLayer)
  * Tensorflow 0.11.0버전에 최적화되어있습니다.
  * 음성파일의 뒷부분 1초를 잘라내어 학습합니다.
  

* Submodule포함해서 받아야함!
  * git clone --recursive https://github.com/pjo901018/GanadaProject  
  * submodule 1. 음성파일을 numpy데이터로 바꿔주는 Tensorflow_DataConverter를 사용했습니다.
    * https://github.com/pjo901018/Tensorflow_DataConverter
  * submodule 2. FTP에서 데이터를 다운로드해주는 FTP_Manager를 사용했습니다.
    * https://github.com/pjo901018/FTP_Manager


* 문제점(v0.1)
  * 턱없이 부족한 트레이닝데이터 개수(42개) 때문에 과적합되었습니다.
  * 음성파일의 뒤 1초를 잘라내어 사용해서 뒤 1초에 실제음성이 없는경우를 고려하지않았습니다.


* Train 결과(42개의 음성(레이블당 3개) 사용)
  ![train](https://raw.githubusercontent.com/pjo901018/GanadaProject/master/image_for_readme/v0.1/train.png)
* train 음성중 하나의 MFCC Feature(실제음성은 0.5~1.2초정도에 존재하는것으로 보임, 파일 맨뒤 1초를 잘라내면 대부분의 음성이 잘려나감) 
  ![MFCC sample](https://raw.githubusercontent.com/pjo901018/GanadaProject/master/image_for_readme/v0.1/mfcc_sample.png)

