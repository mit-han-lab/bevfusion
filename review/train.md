## **BEV Fusion [Code](https://github.com/mit-han-lab/bevfusion/blob/main/tools/train.py) Review**
@author JeongEun Heo

> 
> ```python
> import argparse
> import copy
> import os
> import random
> import time
> 
> import numpy as np
> import torch
> from mmcv import Config
> from torchpack import distributed as dist
> from torchpack.environ import auto_set_run_dir, set_run_dir
> from torchpack.utils.config import configs
> 
> from mmdet3d.apis import train_model
> from mmdet3d.datasets import build_dataset
> from mmdet3d.models import build_model
> from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
> ```
> import → 라이브러리, 모듈


> ```python
> def main():
> 		# distribute training은 여러 대의 GPU를 사용해 모델 학습하는 기술
> 		# dist.init() 함수는 PyTorch에서 제공하는 torch.distributed 패키지를 초기화하여 분산 학습 환경을 구성하는 역할
>     # dist.init() 함수를 호출하면, PyTorch는 현재 노드(노드는 여러 개의 GPU 또는 컴퓨터로 구성된 분산 시스템의 한 단위)의 정보를 설정하고, 분산 학습을 위한 통신 채널을 설정
> 		# 이후 모델과 데이터를 분산하여 학습하거나 테스트할 때, PyTorch는 이 설정된 통신 채널을 사용하여 장치들 간에 데이터를 동기화하고, 모델 파라미터를 전달하여 학습을 진행
> 		dist.init()
> ```
> dist.init() 함수는 PyTorch에서 제공하는 torch.distributed 패키지를 초기화하여 분산 학습 환경을 구성하는 역할


> ```python
> 		# parser 객체를 생성해 명령행 인수를 파싱할 준비
>     parser = argparse.ArgumentParser()
> 		# add_argument() 메서드를 사용해 어떤 인수들을 프로그램에서 받을지 정의한다. 
>     parser.add_argument("config", metavar="FILE", help="config file")
> 		# --가 붙었는데 required를 지정하지 않았으므로 선택적 인수
>     parser.add_argument("--run-dir", metavar="DIR", help="run directory")
>     # 명령행 인수를 파싱하는 메서드
> 		# args : 객체를 사용하여 파싱된 인수를 담은 객체 
> 		# opts : 파싱하지 못한 나머지 인수들을 opt 리스트로 반환한다. 
> 		
> 		# parse_known_args() 사용 목적
> 		# 1. 추가적인 처리가 필요한 알 수 없는 인수들을 opts 리스트에서 확인할 수 있다. 
> 		# 2. 다른 파서들을 함께 사용하는 경우, 
> 		# 하나의 프로그램에서 여러 개의 argparse.ArgumentParser 객체를 사용해 인수를 구하는 경우가 있을 수 있다. 
> 		# 다른 파서들과 함께 사용할 때 불필요한 인수들은 무시하고 남은 인수만 처리할 수 있다. 
> 		args, opts = parser.parse_known_args()
> ```
> parser 객체를 생성해 명령행 인수를 파싱


> ```python
> 		# mmcv 라이브러리의 Config를 사용해 설정 파일을 로드
> 		# arg.config에 해당하는 파일을 읽고, opts에 해당하는 인수를 업데이트한다. 
>     configs.load(args.config, recursive=True)
>     configs.update(opts)
> 
> 		# mmcv.Config를 사용해 설정 정보를 구성하는 cfg 객체를 생성
> 		# recursive_eval() 함수는 설정 파일에서 사용하는 변수를 평가해 실제 값을 구성하는 함수,
> 		# 파일에 사용된 변수들은 실제 값을 대체하여 cfg 객체에 저장한다. 
>     cfg = Config(recursive_eval(configs), filename=args.config)
> ```
> Config를 사용해 설정 파일을 로드하고 설정 정보를 구성하는 cfg 객체를 생성하고 파일에 사용된 변수들에 실제 값을 대체해 cfg 객체에 저장한다. 


> ```python
> 		# cfg.cudnn_benchmark : PyTorch의 CUDA 연산 최적화 기능인 CuDNN Benchmark를 설정,
> 		# 이를 통해 입력 크기에 대한 최적화 알고리즘을 자동으로 선택해 실행 시간을 최적화한다. 
>     torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
>     # 현재 프로세스의 GPU 장치를 설정하는 부분
> 		# dist.local_rank()는 현재 노드의 로컬 GPU 장치의 인덱스를 반환한다. 
> 		# 즉 현재 프로세스가 할당된 GPU를 설정한다. 
> 		torch.cuda.set_device(dist.local_rank())
> ```
> CuDNN Benchmark 설정, 현재 프로세스의 GPU 장치를 설정 


> ```python
> 		# args.run_dir가 None인 경우 
> 		# auto_set_run_dir()를 호출해 자동으로 실행 디렉토리를 설정한다. 
> 		# auto_set_run_dir() 함수는 현재 날짜와 시간을 기준으로 실행 디렉토리를 생성하고 반환
> 		# 실행 디렉토리는 프로그램의 실행 결과와 로그 등을 저장하는 디렉토리이다. 
>     if args.run_dir is None:
>         args.run_dir = auto_set_run_dir()
> 		# Node이 아닌 경우 사용자가 명령행에서 --run-dir을 지정한 것이므로, 
> 		# 해당 값으로 실행 디렉토리를 설정한다. 
>     else:
>         set_run_dir(args.run_dir)
> 		# cfg.run_dir 값을 args.run_dir로 설정해 설정 객체 cfg에 실행 디렉토리 정보를 저장
> 		# 이를 통해 프로그램이 실행되는 동안 해당 실행 디렉토리에 결과 파일이나 로그를 저장하게 된다.
>     cfg.run_dir = args.run_dir
> ```
> 실행 디렉토리 설정 부분


> ```python
> 		# 설정 정보들이 담긴 cfg 객체의 내용을 configs.yaml이라는 파일로 실행 디렉토리에 저장하는 것을 의미한다. 
> 		# cfg : mmcv.Config 클래스로부터 생성된 설정 정보들이 담긴 객체 
> 		# dump() : 해당 객체에 저장된 설정 정보들을 파일로 저장하는 메서드
> 		# os.path.join(cfg.run_dir, "configs.yaml") : 실행 디렉토리인 cfg.run_dir과 파일 이름 configs.yaml을 합쳐 파일의 전체 경로를 생성
> 		# os.path.join() : 여러 경로를 하나로 결합해 전체 경로를 생성하는 함수 
> 		# dump config
>     cfg.dump(os.path.join(cfg.run_dir, "configs.yaml")
> ```
> 설정 정보들이 담긴 cfg 객체의 내용을 configs.yaml이라는 파일로 실행 디렉토리에 저장하는 것


> ```python
> 		# 현재 시간을 기준으로 로그 파일 이름을 생성, get_root_logger() 함수를 호출해 로거를 생성한다. 
> 		# 로그 파일은 실행 디렉토리에 생성되며, 프로그램의 실행 과정과 중요한 정보들이 기록된다. 
>     # init the logger before other steps
>     timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
>     log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
>     logger = get_root_logger(log_file=log_file)
> ```
> 현재 시간을 기준으로 로그 파일 이름, get_root_logger() 함수를 호출해 로거를 생성


> ```python
> 		# logger : 로깅을 담당하는 객체, get_root_logger()를 통해 생성되었다. 
> 		# 이 객체를 사용해 로그 메시지를 출력할 수 있다. 
> 		# logger.info() : 로그 레벨이 info인 메시지를 출력하는 메서드로 info 레벨은 실행 중 중요한 정보를 출력할 때 사용된다. 
> 		# f"Config:\n{cfg.pretty_text} : 출력할 로그 메시지의 내용을 포매팅하는 부분, 
> 		# cfg.pretty_text 속성을 사용해 설정 정보들을 문자열 형태로 반환하고, 앞에 Config:라는 문자열을 붙여 출력한다. 
> 		# cfg.pretty_text에는 설정 정보들이 포맷팅 되어 문자열 형태로 들어있다. 
>     # log some basic info
>     logger.info(f"Config:\n{cfg.pretty_text}")
> ```
> 로그 레벨이 info인 메시지를 출력


> ```python
> 		# cfg.seed == 랜덤 시드 값, 난수 생성을 위해 사용되는 초기값으로 동일한 랜덤 시드를 사용하면 같은 순서로 랜덤한 값을 생성할 수 있다. 
> 		# 프로그램의 실행 결과를 재현하고 실험의 인관성과 재현성을 보장할 수 있다.  
> 		# cfg.seed 값이 None이 아닌 경우 == 설정 파일에서 랜덤 시드를 지정한 경우에만 해당 조건을 만족한다. 
> 		# set random seeds
>     if cfg.seed is not None:
>         logger.info(
>             f"Set random seed to {cfg.seed}, "
>             f"deterministic mode: {cfg.deterministic}"
>         )
> 				# 랜덤 시드를 여러 번 설정하는 이유 : 아래 3가지 랜덤 모듈이 각각 독립적으로 랜덤한 숫자를 생성하기 때문이다. 
> 				# 한번만 랜덤 시드를 설정하면 동일한 랜덤 순서를 가지게 된다. 
> 				# 이로 인해 3가지 모듈의 랜덤 함수들이 같은 순서로 랜덤한 숫자를 생성하게 되므로 일종의 상관성이 생길 수 있다. 
> 				# 이를 방지하기 위해 3가지 모듈의 랜덤 시드를 별도로 설정해 독립적으로 랜덤 순서를 유지하는 것이 중요하다. 
> 				
> 				# 랜덤 시드 설정 : Python 기본 모듈 random을 사용
>         random.seed(cfg.seed)
> 				# 랜덤 시드 설정 : NumPy 랜덤 모듈 np.random을 사용 
>         np.random.seed(cfg.seed)
> 				# 랜덤 시드 설정 : PyTorch의 랜덤 모듈 torch를 이용
>         torch.manual_seed(cfg.seed)
>         # cfg.deterministic : 설정 파일에서 지정한 값으로 PyTorch CUDA 연산 최적화인 CuDDNN을 제어하는 옵션
> 				# True인 경우 : CuDDNN의 랜덤 알고리즘을 끄고, 동일한 입력에 대해 동일한 출력을 생성
> 				# 학습 과정이나 모델 평가 과정에서 같은 입력에 대해서 항상 같은 결과를 얻을 수 있다.
> 				if cfg.deterministic:
> 						# 하지만 torch.backends.cudnn.deterministic = True의 경우 성능 저하를 일으킬 수 있으므로
> 						# 학습 속도가 중요한 경우 설정하지 않는 것이 좋다. 
>             torch.backends.cudnn.deterministic = True
> 						# CUDA 연산 최적화를 위해 CuDNN 라이브러리의 벤치마크를 사용하지 않도록 설정하는 부분,
> 						# 학습 과정에서 CUDA 연산의 최적화 알고리즘이 사용되지 않고, 고정된 알고리즘이 사용된다. 
>             torch.backends.cudnn.benchmark = False
> ```
> 랜덤 시드 값 설정


> ```python
> 		# cfg.data.train의 정보를 이용해 데이터셋을 빌드해 datasets 리스트에 저장한다. 
> 		# build_dataset() : mmdet3d.models 모듈에서 가져온 함수로, 설정 파일 cig에 정의된 데이터셋을 구성하여 데이터를 준비 
> 		# 데이터셋은 훈련에 사용할 이미지와 라벨을 포함하는 객체
>     datasets = [build_dataset(cfg.data.train)]
> ```
> cfg.data.train의 정보를 이용해 데이터셋을 빌드해 datasets 리스트에 저장


> ```python
> 		# cfg.model 정보를 이용해 모델을 빌드해 model 변수에 저장
> 		# build_model() : mmdet3d.models 모듈에서 가져온 함수로, 설정 파일(cfg)에 정의된 모델 아키텍처를 구성하여 모델을 생성
> 		# 모델은 신경망 아키텍처와 파라미터들을 정의하는 객체
>     model = build_model(cfg.model)
> ```
> cfg.model 정보를 이용해 모델을 빌드해 model 변수에 저장


> ```python
> 		# init_weights() 함수를 호출해 모델의 가중치를 초기화
> 		# 모델 객체 model에 정의된 가중치를 초기화하는 메서드
> 		# 모델을 처음 생성한 후에는 가중치들이 무작위로 초기화 되어 있는데, 
> 		# 이를 특정 방법으로 초기화해 모델의 학습을 원활하게 시작할 수 있도록 한다. 
>     model.init_weights()
> ```
> init_weights() 함수를 호출해 모델의 가중치를 초기화


> ```python
> 		# cfg.get("sync_bn", None) : cfg 딕셔너리에서 sync_bn 키에 해당하는 값을 가져와 확인, 만약 키가 존재하지 않는다면 기본값으로 None으로 반환한다. 
> 		if cfg.get("sync_bn", None):
> 				# sync_bn 값이 딕셔너리 타입이 아니라면
>         if not isinstance(cfg["sync_bn"], dict):
> 						# sync_bn 값이 딕셔너리 타입이 아니라면, cfg["sync_bn"]을 빈 딕셔너리로 초기화 
>             # dict(exclude=[]) : 빈 딕셔너리를 생성하는 표현식 
> 						cfg["sync_bn"] = dict(exclude=[])
> 				# sync_bn이 딕셔너리 타입인 경우 모델에 Sync Batch Normalization을 적용
> 				# Sync Batch Normalization : 분산 학습 환경에서 배치 정규화를 적용하는 기법
> 				# convert_sync_batchnorm() : mmdet3d.utils 모듈에서 가져온 함수이다. 
> 				# model : build_model(cfg.model)에서 생성한 모델이 전달된다. 인자 값은 cfg["sync_bn"]["exclude"]이다. 
> 				# 이 함수가 모델에 Sync Batch Normalization을 적용한다. 
> 				# exclude=cfg["sync_bn"]["exclude"] : 배치 정규화를 적용하지 않을 레이어들의 이름을 담고 있는 리스트 
>         model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])
> ```
> init_weights() 함수를 호출해 모델의 가중치를 초기화


> ```python
> 		# 로거를 사용해 모델 정보를 출력
>     logger.info(f"Model:\n{model}")
> ```
> 로거를 사용해 모델 정보를 출력


> ```python
> 		# train_model() 함수를 호출해 모델 훈련
> 		# train_model() : mmdet3d.apis 모듈에서 가져온 함수로 주어진 model을 주어진 datasets으로 훈련시키는 역할을 한다. 
> 		# model : 훈련할 모델 객체
> 		# dataset : 훈련에 사용할 데이터셋을 담고 있는 리스트, 이전에 datasets = [build_dataset(cfg.data.train)]에서 생성한 데이터셋이 사용된다. 
> 		# cfg : 설정 파일을 담고 있는 Config rorcp 
> 		# distributed=True : 분산 학습을 사용할지 여부를 나타내는 옵션
> 		# validate=True : 훈련 중 검증을 수행할지 여부를 나타내는 옵션
> 		# timestamp=timestamp : 현재 시간 정보를 나타내는 timestamp 변수를 인자로 전달 
>     train_model(
>         model,
>         datasets,
>         cfg,
>         distributed=True,
>         validate=True,
>         timestamp=timestamp,
>     )
> ```
> train_model() 함수를 호출해 모델 훈련


> ```python
> if __name__ == "__main__":
>     main()
> ```
> main() 함수 호출