## 1. 평가지표

### 1.1 혼동행렬

<table>
    <tr>
        <td></td>
        <td colspan="3" align="center">실제 정답</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">분류결과</td>
        <td></td>
        <td>True</td>
        <td>False</td>
    </tr>
    <tr>
        <td>Accept</td>
        <td>1. TP</td>
        <td>2. FP</td>
    </tr>
    <tr>
        <td>Reject</td>
        <td>3. FN</td>
        <td>4. TN</td>
    </tr>
</table>

- 1. TP
- 2. FP
- 3. FN
- 4. TN

## 2. GT 정의

### 2.1 쿼리 하나를 정한다

- 각 FAISS결과 JSON 파일 이름에서 쿼리 이미지 ID를 뽑는다.\
    e.g. `..._251124160703_00031_....json` → `251124160703_00031.jpg`
- 이 쿼리 이미지의 위경도/좌표는 `jamshill_flight_index.json`에서 읽어 온다.

### 2.2 참조 전체와 거리계산

- 참조 인덱스 `jamshill_reference_index.json`에는 모든 참조 이미지의 위경도/좌표가 들어 있다.
- 쿼리 1개와 **참조전체**를 비교해서 거리를 계산한다.

    - geo 기준: Haversine 거리

    $$

    $$

### 2.3 "가장 가까운 1개"를 GT로 고정

-  참조 전체 중 거리 최소값을 가진 참조 이미지 1개만 GT로 선택한다.
- 코드에선 `_min_distance_to_refs(...)`가 그 역할

```python
gt_geo_ref, gt_geo_dist = _min_distance_to_refs(query_meta, ref_items, mode="geo")
gt_xy_ref, gt_xy_dist = _min_distance_to_refs(query_meta, ref_items, mode="xy")
```
- 여기서 `gt_geo_ref` 또는 `gt_xy_ref`가 단일 GT.

### 2.4 TopK 결과와 비교

- FAISS가 뽑은 TopK 후보 목록에서 GT가 포함되면 Recall@K=1, 아니면 0.

```python
geo_rank = rank_of(gt_geo_ref)  # GT가 몇 번째인지
geo_recall = 1 if geo_rank is not None else 0
```
## 3. 성능지표

### 3.1 Recall@K

의미: *"정답(또는 정답 집합)이 Top-K 안에 포함되어 있는가?"*
> **단일 정답** 
$$
\text{Recall@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{\text{GT}_i} \in \mathbf{\text{TopK}_i}
$$
    정답이 TopK에 있으면 1, 없으면 0

> **여러 정답**
$$
\text{Recall@K} = \frac{1}{N}\sum_{i=1}^{N}\frac{\mathbf{|\text{TopK}_i \cap \text{GT}_i|}}{\mathbf{|\text{GT}_i|}}
$$
    정답 중 몇 개를 TopK에서 찾았는지 비율
---

### 3.2 Precision@K

의미: *"Top-K안의 결과 중 실제 정답이 얼마나 섞여 있는가?"*

> **단일정답**
$$
\text{Precision@K} = \frac{1}{N}\sum_{i=1}^{N}\frac{\mathbf{\text{GT}_i \in \text{TopK}_i}}{\mathbf{K}}
$$
    정답이 TopK에 있으면 1/K, 없으면 0

> **여러 정답**
$$
\text{Precision@K} = \frac{1}{N}\sum_{i=1}^{N}\frac{\mathbf{|\text{TopK}_i \cap \text{GT}_i|}}{\mathbf{K}}
$$
    TopK 중 정답 비율
---

### 3.3 Hit@K

의미: *"Top-K 안에 정답이 하나라도 있는가?"*
$$
\text{Hit@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{\text{TopK}_i}\cap \mathbf{\text{GT}_i} \neq \emptyset
$$
    사실상 Recall@K(단일 정답)과 동일한 개념으로 쓰인다.
---
