# FAISS

## 색인 기법

### 1. Flat (Brute-Force) 계열

- IndexFlatL2 / IndexFlatIP
  
- 원리: 모든 벡터와의 거리를 직접 계산 (L2 거리 또는 Inner Product)  
  특징:   
  - 근사 아님 -> 정확도 100%  
  - 인덱싱 (색인) 과정 없음, 추가 벡터 즉시 삽입 가능  
  - 데이터가 많을 수록 검색 시간이 선형적으로 증가  
  장점: 단순하고 정확도 보장  
  단점: 대규모 데이터셋에서 속도가 느림  
  사용 사례: 데이터 규모가 작거나 정확도가 최우선인 경우
  
### 2. IVF (Inverted File) 계열

- IndexIVFFlat / IndexIVFPQ / IndexIVFSQ
  
- 원리:  
  - 벡터 공간을 k-means로 미리 클러스터링하여 여러 Voronoi Cell (Centroid)로 나눔.  
  - 검색 시 쿼리 벡터가 속할 가능성이 높은 nprobe 개의 셀만 탐색.  
  특징:  
  - nlist: 클러스터 수  
  - nprobe: 검색 시 탐색할 클러스터 수 (정확도/속도 트레이드오프)  
  세부변형:  
  - IVFFlat: 클러스터 내 벡터를 원본 그대로 저장  
  - IVFPQ: 클러스터 내 벡터를 Product Quantization으로 압축 저장 -> 메모리 절약  
  - IVFSQ: Scalar Quantization 사용  
  장점: 대규모 데이터에서 매우 빠름  
  단점: 초기 학습 (K-Means)필요. 파라미터 튜닝 필요.  
  사용 사례: 수백만 ~ 수억 벡터 검색, 검색 정확도보다 속도가 중요한 경우.
  
### 3. PQ (Product Quantization) 계열

- IndexPQ / IndexIVFPQ
  
- 원리  
  - 벡터를 여러 서브 벡터로 나누고, 각 부분을 별도의 코드북(Codebook)으로 양자화(Quantization)  
  - 원본 대신 양자화된 코드로 근사 거리 계산.  
  특징:  
  - m: 서브벡터 수  
  - nbits: 각 서브벡터 양자화 비트 수  
  장점: 메모리 사용량 크게 절감  
  단점: 정확도는 Flat보다 낮음  
  사용사례: 메모리가 제한된 환경에서 대규모 데이터 검색
  
### 4. HNSW (Hierarchical Navigable Small World Graph)

- IndexHNSWFlat
  
- 원리:  
  - 그래프 기반 ANN 알고리즘  
  - 계층적 그래프 레이어를 만들고 탐색시 고차원에서 저차원으로 점차 근사 최근접을 찾아감.  
  특징:  
  - M: 각 노드의 최대 연결 수  
  - efSearch: 검색 시 탐색 폭  
  장점: 매우 높은 검색 정확도, 빠른 검색  
  단점: 메모리 사용량이 비교적 큼  
  사용사례: 높은 정확도와 빠른 검색을 동시에 원하는 서비스
  
### 5. 기타 특수 기법

- LSH (Locality Sensitive Hashing)
  
    - IndexLSH
      
        - 해시 기반 근사 검색
          
        - 고차원에서 거리 보존 해시를 통해 빠른 근사 검색
          
- Hierarchical K-Means (HKMeans)
  
    - IndexIVF with hierarchical k-means:
      
        - IVF의 확장형으로 계층적 클러스터링을 사용.
          
- GPU 인덱스
  
    - 모든 주요 색인 기법은 GPU 가속 버전(GpuIndexFlatL2, GpuIndexIVFPQ 등) 제공.
      
### 색인 기법 비교 요약

-   
  1) 기법: Flat  
  정확도: 최고  
  속도: 낮음  
  메모리: 큼  
  특징: 정확도 100%, 소규모 데이터  
    
  2) 기법: IVF  
  정확도: 중간~높음  
  속도: 빠름  
  메모리: 중간  
  특징: 클러스터링 기반, 대규모 데이터  
    
  3) 기법: PQ  
  정확도: 중간  
  속도: 빠름  
  메모리: 낮음  
  특징: 메모리 절약형 근사  
    
  4) 기법: IVFPQ  
  정확도: 중간  
  속도: 빠름  
  메모리: 낮음  
  특징: IVF + PQ 결합  
    
  5) 기법: HNSW  
  정확도: 높음  
  속도: 매우 빠름  
  메모리: 큼  
  특징: 그래프 기반 고정밀 ANN
  
* FAISS는 Flat → IVF → PQ → HNSW 등 다양한 색인 전략을 제공하며,
정확도 우선: IndexFlatL2, IndexHNSWFlat  
대규모/빠른 검색: IndexIVFPQ  
메모리 절약: IndexPQ, IndexIVFPQ  
등으로 요구사항에 따라 선택할 수 있다.
      
## 색인기법+PCA

### FAISS에서 PCA(Principal Component Analysis) 기반 차원 축소는 특정 색인(index) 타입에만 국한되지 않고, 모든 색인 앞단에 “전처리(Pre-transform)” 단계로 결합하

### , PCA는 “인덱싱 기법” 그 자체가 아니라 임베딩 벡터의 차원을 줄이기 위해 색인 앞단에 붙일 수 있는 Transform입

### FAISS가 이를 위해 제공하는 핵심 도구는 IndexPreTransform 과 PCAMatrix

### 1. 적용 방식

- ① PCAMatrix + IndexPreTransform  
    
  PCAMatrix: FAISS에서 PCA를 수행하기 위한 모듈  
  훈련 데이터로 주성분을 학습한 뒤, 지정한 차원으로 변환  
  예: PCAMatrix(original_dim, target_dim)  
    
  IndexPreTransform: 색인 앞에 전처리 단계를 끼워넣는 래퍼(wrapper)  
  첫 번째 단계로 PCAMatrix를 등록하고,  
  두 번째 단계로 원하는 임의의 색인(Flat, IVF, HNSW 등)을 등록.
  
### 2. PCA를 함께 쓸 수 있는 주요 색인 예시

- PCA는 아래와 같이 거의 모든 주요 색인 기법과 조합 가능.  
  1) 색인방식: IndexFlatL2 / IndexIVFPQ  
  PCA 결합 방법: IndexPreTransform(PCAMatrix, IndexFlatL2)  
  특징: 정확도 100% brute-force + 차원 축소  
    
  2) 색인방식: IndexIVFFlat / IndexIVFPQ  
  PCA 결합 방법: IndexPreTrainsform(PCAMatrix, IndexIVPFFlat)  
  특징: IVF클러스터링 전 차원 축소로 검색 속도 및 메모리 절감  
    
  3) 색인방식: IndexPQ / IndexIVFPQ  
  PCA 결합 방법: IndexPreTransform(PCAMatrix, IndexIVFPQ)  
  특징: PQ 전에 차원 축소 -> 양자화 오류를 줄여 성능 향상  
    
  4) 색인방식: IndexLSH  
  PCA 결합 방법: IndexPreTransform(PCAMatrix, IndexLSH)  
  특징: LSH 해싱 전에 차원 축소  
    
  * 즉, Flat-IVF-PQ-HNSW 등 모든 대표 인덱스에 PCA를 추가 가능
  
### 3. 왜 PCA를 적용하나?

- 차원 축소: 예를 들어 768차원 임베딩 → 128차원으로 축소 시, 검색 속도 향상·메모리 절감.  
- 노이즈 제거: 데이터의 주요 분산 방향만 남기므로 검색 품질이 오히려 좋아질 수 있음.  
- 후처리/인덱싱 부담 감소: 특히 IVF, PQ 같은 근사 검색 기법의 학습·탐색 속도 개선.
  
### 4. 주의사항

- PCA는 훈련(Training)이 필요: PCAMatrix.train(training_vectors)을 통해 주성분을 학습해야 함. 차원을 너무 많이 줄이면 정보 손실로 검색 정확도가 떨어질 수 있음. GPU 버전(GpuIndexPreTransform)에서도 동일하게 사용 가능.
  
### 요약

## 임베딩 방식

### 1. 전통적 특징 추출(Pre-Deep Learning)

- FAISS에서 PCA는 색인 기법의 일부가 아니라,  
  PCAMatrix를 IndexPreTransform과 함께 사용하여 Flat, IVF, PQ, HNSW 등 모든 인덱스 앞단에 적용할 수 있는 차원 축소 단계.  
  따라서 “PCA를 적용할 수 있는 색인 방식” = 거의 모든 FAISS 색인 방식이며, 대표적으로 IndexFlatL2, IndexIVFPQ, IndexHNSWFlat 등이 있다.
  
- 전통적 특징 추출 기법별 특징, 장점, 한계 요약   
    
  1) SIFT (Scale-Invariant Feature Transform)  
  특징: 스케일/회전에 불변인 키포인트 추출  
  장점: 강인한 로컬 특징  
  한계: 고차원 특징, 연산량 큼  
    
  2) SURF (Speeded Up Robust Features)  
  특징: SIFT 개선, 계산 속도 향상  
  장점: 빠른 처리  
  한계: 특허 문제로 제약  
    
  3) ORB (Oriented FAST and Rotated BRIEF)  
  특징: FAST+BRIEF 결합, 회전 불변  
  장점: 오픈소스, 실시간 처리  
  한계: 복잡한 이미지에서 성능 제한  
    
  4) HOG (Histogram of Oriented Gradients)  
  특징: 그래디언트 방향 히스토그램  
  장점: 보행자 인식 등 전통적 CV  
  한계: 복잡한 시각 패턴 표현 한계  
    
  * 딥러닝 이전에는 이미지의 시각적 특징을 **수학적 기술자(feature descriptor)**로 표현했음.
  
### 2. CNN 기반 딥러닝 임베딩

- CNN 기반 딥러닝 임베딩 모델 방식 별 특징 및 활용 요약  
    
  1. ImageNet Pretrained CNN (ResNet, VGG, Inception 등)  
  특징: 마지막 FC Layer 이전의 feature map을 벡터로 사용  
  활용: 이미지 검색, 분류  
    
  2. ResNet50/101  
  특징: Residual Connection으로 깊은 네트워크 학습  
  활용: 범용 임베딩  
    
  3. DenseNet, MobileNet  
  특징: 메모리/속도 최적화  
  활용: Edge device 임베딩  
    
  -> 보통 Global Average Pooling (GAP) 후 얻은 feature vector를 그대로 임베딩으로 사용.  
    
  * Convolutional Neural Networks를 활용해 이미지의 고수준 특징을 자동 추출.
  
### ? ViT

- Vision Transformer(ViT) 계열 모델은 이미지 임베딩 방법을 분류했을 때, “Self-Supervised & Representation Learning” 계열과 “멀티모달(Vision-Language)” 계열 모두에서 중요한 축을 차지.  
  왜냐하면 ViT는 **모델 아키텍처(Transformer)**를 의미하는 것이고, 그 위에 어떤 학습 전략을 얹느냐에 따라 사용 분야가 달라지기 때문.
  
- 1. ViT의 기본 개념  
  ViT (Vision Transformer): 2020년 Google Research에서 발표.  
  CNN 대신 Transformer 인코더를 이미지 패치에 적용:  
  이미지를 일정 크기의 패치(patch)로 나눈 뒤, 각 패치를 토큰으로 간주하고 Transformer로 처리.  
  CNN보다 **전역적 문맥(global context)**을 잘 포착.  
  ➡ ViT 자체는 임베딩을 추출하는 모델 구조이며, 학습 방식에 따라 여러 계열로 분류 가능.
  
- 2. 분류 상의 위치 (분류 별 ViT의 대표 모델 및 설명)  
  1) Self-Supervised & Representation Learning  
  ViT의 대표 모델: DINO, DINOv2, MAE (Masked Autoencoder), MoCo v3, BEiT  
  설명: 라벨 없이 대규모 데이터에서 범용 이미지 표현을 학습. ViT를 Backbone으로 사용해 높은 성능의 범용 임베딩 제공  
    
  2) 멀티모달 (Vision-Language)  
  ViT의 대표 모델: CLIP (OpenAI), ALIGN, BLIP, Flamingo  
  설명: ViT가 이미지 인코더로 사용되어 텍스트 임베딩과 공통 latent space를 구축. 텍스트/이미지 검색/RAG 등에 활용됨.  
    
  3) 기타 지도학습(Classification)  
  ViT의 대표 모델: 원본 ViT, DeiT  
  설명: ImageNet 등 라벨된 데이터셋으로 지도 학습, CNN 대체.  
    
  * 즉, ViT는 아키텍쳐이며, Self-Supervised/멀티모달 등 학습 패러다임에 따라 다른 카테고리에 속함.
  
- 3. ViT 계열의 대표 활용 방식  
  ① 순수 비전(Self-Supervised) 임베딩  
    
  DINO/DINOv2: Self-distillation, Contrastive 학습으로 라벨 없이 강력한 이미지 표현 학습.  
    
  MAE: Masked Autoencoder로 이미지 일부를 마스킹 후 복원.  
    
  BEiT: BERT 스타일 마스크드 이미지 모델링.  
    
  ➡ 이 계열은 “Self-Supervised & Representation Learning” 카테고리에 속함.  
    
  ② 텍스트-이미지 멀티모달 임베딩  
    
  CLIP: ViT + Text Transformer → 이미지와 텍스트를 동일한 임베딩 공간으로 매핑.  
    
  BLIP, BLIP-2: 비전-언어 이해 및 이미지 캡셔닝.  
    
  ALIGN: 대규모 웹 이미지-텍스트 데이터 기반.  
    
  ➡ 이 계열은 “멀티모달(Vision-Language)” 카테고리에 속함.
  
- 4. 선택 가이드  
    
  범용 이미지 검색·분류: DINOv2, MAE 기반 ViT  
    
  텍스트-이미지 검색·RAG: CLIP, BLIP, ALIGN  
    
  라벨 있는 지도학습 분류: ViT/DeiT 원본
  
- 어떤 학습 패러다임을 쓰느냐에 따라 “Self-Supervised Representation Learning”과 “멀티모달” 두 카테고리 모두에 속한다고 볼 수 있다
  
### 3. Metric Learning 기반 임베딩

- Metric Learning 기반 임베딩 방법별 특징 및 예시 요약  
    
  1) Siamese Network  
  특징: 두 이미지를 입력으로 받아 거리 학습  
  예시: FaceNet (얼굴 임베딩)  
    
  2) Triplet Loss  
  특징: Anchor, Positive, Negative 샘플을 사용  
  예시: FaceNet, DeepFace  
    
  3) Contrastive Loss  
  특징: 쌍(Pair) 이미지 간 거리를 최소 / 최대화  
  예시: 시각적 유사도 검색  
    
  * 얼굴 인식(FaceNet, ArcFace), 이미지 검색, Re-ID (Re-Identification)에 널리 사용.  
  * 임베딩 공간에서 유사한 이미지끼리 가깝게, 다른 클래스는 멀리 떨어지도록 학습.
  
### 4. Self-Supervised & Representation Learning

- Self-Supervised & Representation Learning 모델별 특징 및 활용 요약  
  1) SimCLR, BYOL, MoCo  
  특징: Constrasive Learning 기반  
  활용: 대규모 비지도 학습  
    
  2) SwAV  
  특징: 클러스터링 기반 self-supervised  
  활용: 범용 이미지 표현  
    
  3) DINO, DINOv2 (Meta)  
  특징: ViT 기반 Self-Distillation  
  활용: 최신 범용 이미지 표현  
    
  * 사전학습된 모델을 가져와 임베딩 추출 시 라벨 데이터 불필요.  
  * 라벨 없이 대규모 데이터에서 표현을 학습해 범용 임베딩 제공.
  
### 5. 멀티모달 & Vision-Language 모델

- 멀티모달 & Vision-Language 모델 별 특징 및 활용 요약  
    
  1) CLIP (OpenAI)  
  특징: 이미지/텍스트 쌍 학습, 공통 Latent Space  
  활용: 텍스트 기반 이미지 검색  
    
  2) BLIP/BLIP-2  
  특징: 이미지 캡셔닝과 멀티모달 이해  
  활용: Visual Question Answering  
    
  3) ALIGN, Florence  
  특징: 대규모 웹 이미지-텍스트 데이터로 학습  
  활용: 범용 이미지/텍스트 검색  
    
  4) SigLIP (Google)  
  특징: CLIP 개선, 효율적 학습  
  활용: 대규모 멀티모달 검색  
    
  * 최근 RAG(Retrieval-Augmented Generation), 멀티모달 LLM, 이미지 검색 엔진에 핵심.  
  * 텍스트와 이미지를 공통 임베딩 공간에 매핑해, 텍스트-이미지 검색이나 생성형 AI에서 핵심.
  
### 6. 특수 도메인 임베딩

- 얼굴 인식: FaceNet, ArcFace, DeepFace  
    
  의료 영상: BioMedCLIP, MedCLIP  
    
  위성/항공 이미지: SatCLIP, SEN12MS 모델  
    
  상품/패션 이미지: DeepFashion embedding 모델
  
### 7. 선택 가이드

- 범용 검색/추천: CLIP, DINOv2, SimCLR
  
- 텍스트-이미지 검색: CLIP, BLIP
  
- 얼굴 인식/바이오메트릭스: FaceNet, ArcFace
  
- 모바일/경량: MobileNet, EfficientNet
  
- 대규모 비지도 데이터 활용: BYOL, MoCo, DINO
  
