# VFX_hw2
R11946012 資科所 王奕方 
R10942198 電信丙 林仲偉

## 1. Description
在定點拍攝全景場景不同方向的照片，先使用 feature detection 找出兩兩照片之間的特徵點，然後使用 feature matching 找出 match point pairs，最後根據 match point pairs 做 image stitching，並使用 blendering 使得拼接後影像邊界視覺上看起來沒有接縫，以得到一張全景圖。

## 2. Experiment Setup

| 項目 | 描述                  |
|:---- |:--------------------- |
| 機型 |    SONY ILCE-6400          |
| 焦距 |        43mm          |
| ISO  |       1000             |
| F    |          4.5          |
| 快門時間   |          1/15s          |


### 場景圖片
| 項目   | 描述      |
|:------ |:--------- |
| 尺寸   |   400x600 |
| 解析度 |  72 |
| 數量   | 20        |


* 場景ㄧ：水源會館前院

<img src="https://i.imgur.com/YcTHH4F.jpg" width="200px"><img src="https://i.imgur.com/2cdx7Mb.jpg" width="200px"><img src="https://i.imgur.com/A4tJUP6.jpg" width="200px"><img src="https://i.imgur.com/XuGh8AX.jpg" width="200px"><img src="https://i.imgur.com/C9L4TYF.jpg" width="200px"><img src="https://i.imgur.com/ctXEbej.jpg" width="200px"><img src="https://i.imgur.com/PZ9fVN7.jpg" width="200px"><img src="https://i.imgur.com/chZOxN4.jpg" width="200px"><img src="https://i.imgur.com/LZq1xG9.jpg" width="200px"><img src="https://i.imgur.com/aziLJpl.jpg" width="200px"><img src="https://i.imgur.com/u5AV2A5.jpg" width="200px"><img src="https://i.imgur.com/3J5ZA9Q.jpg" width="200px"><img src="https://i.imgur.com/36IXExP.jpg" width="200px"><img src="https://i.imgur.com/XvdCesB.jpg" width="200px"><img src="https://i.imgur.com/RRlrtxM.jpg" width="200px"><img src="https://i.imgur.com/WnqBNdF.jpg" width="200px"><img src="https://i.imgur.com/GmeeQIe.jpg" width="200px"><img src="https://i.imgur.com/ZcvRQaD.jpg" width="200px"><img src="https://i.imgur.com/8RrGOfI.jpg" width="200px"><img src="https://i.imgur.com/YYy9uNb.jpg" width="200px">


* 場景二：水源會館草皮

<img src="https://i.imgur.com/YK1o1TQ.jpg" width="200px"><img src="https://i.imgur.com/K9GCiaA.jpg" width="200px"><img src="https://i.imgur.com/skdS66n.jpg" width="200px"><img src="https://i.imgur.com/T6AY1w6.jpg" width="200px"><img src="https://i.imgur.com/Rnbpj6Z.jpg" width="200px"><img src="https://i.imgur.com/L2oDk5N.jpg" width="200px"><img src="https://i.imgur.com/WpaD11s.jpg" width="200px"><img src="https://i.imgur.com/1JVlEP7.jpg" width="200px"><img src="https://i.imgur.com/kt6c68g.jpg" width="200px"><img src="https://i.imgur.com/WvIXFFd.jpg" width="200px"><img src="https://i.imgur.com/FKl4cB4.jpg" width="200px"><img src="https://i.imgur.com/B5jDUQU.jpg" width="200px"><img src="https://i.imgur.com/nnYkWP7.jpg" width="200px"><img src="https://i.imgur.com/QymFXLs.jpg" width="200px"><img src="https://i.imgur.com/1FK3vjl.jpg" width="200px"><img src="https://i.imgur.com/mzczKYr.jpg" width="200px"><img src="https://i.imgur.com/TYSzyMv.jpg" width="200px"><img src="https://i.imgur.com/NrgTIZR.jpg" width="200px"><img src="https://i.imgur.com/yDWNKhO.jpg" width="200px"><img src="https://i.imgur.com/GkWtlST.jpg" width="200px">


## 3. Program Workflow
1. 使用 autostitch 去得到所有照片的 focal length
2. 對照片做 cylindrical projection
3. 使用 Harris Corner Detector 取得 keypoint
4. 使用 SIFT 的方式取得 feature descriptor
5. 使用 brute force 的方式比較 euclidean distance 做 Feature matching
6. 用 RANSAC 去找出使得 Image matching 結果最好的 shift amount, 並依此對兩張圖片做 Image Stitching. (在此作業中，我們假設只會發生平移)
7. 做 Linear Blending
8. 重複 2~6, 直到所有照片都被拼接完成。 

## 4. Implementation Detail

### (1) Cylindrical Projection

### (2) Feature Detection: 
使用 Harris Corner Detector 取得 keypoint:
1. 使用 Gaussian Filter 將灰階圖片平化後，取得x方向和y方向的 gradient $I_{x}$ 和 $I_{y}$
2. 計算 gradient 乘積 $I_{x}^{2}, I_{y}^{2}, I_{x}*I_{y}$
3. 用 Gaussian Filter 作為 window function 計算 gradient 乘積的加總 $S_{x^{2}}, S_{xy}, S_{xy}, S_{y^{2}}$
4. 得到 M 矩陣 
$
M=
\left\(
\begin{matrix}
S_{x^{2}} & S_{xy} \\\
S_{xy} & S_{y^{2}} \\\
\end{matrix}
\right\)
$
5. 計算 corner response $R = detM - k(traceM)^{2}$，這裡 k 值使用 0.04
6. 以 0.01*max(response) 作為 threshold，篩選掉小於 threshold 不為 corner 的 keypoint
7. 使用 maximum_filter 做 nonmax supression，篩選掉太過相近的點
8. 剩下的點即為 keypoints

使用 SIFT descriptor 取得 feature descriptor:  
**orientation assignment**
1. 使用 Gaussian Filter 將灰階圖片平化後，取得x方向和y方向的 gradient
2. 使用 x 和 y 的 gradient 計算 keypoint orientation 的角度 $\Theta$ 和強度 $m$
3. 將 orientation 以每 10 度分成一個 bucket，得到 historgram
4. 以每個 bucket 中間值作為 keypoint 的 orientation，以 keypoint 為中心旋轉
5. 實現 keypoint descriptor orientation invariant

**local image descriptor**
1. 使用同樣的方式，但將 orientation 以每 45 度分成一個 bucket，得到 8 個 orientations 的 historgram
2. 以 keypoint 為中心，取得 16x16 array，並將其分成 4x4 sub-array，每個 sub-array 統計 8 個 orientations 各自次數
3. 以 8 x 4x4 = 128 dimensions 表示為一個 keypoint descriptor
4. Normalize dimensions 來避免光線造成的變化 (>0.2 clip) 

### (3) Feature Matching
1. 以 brute force 兩倆比對兩張圖片間的 features
2. 使用 euclidean distance 計算 features 的相似度
3. 差距小於 threshold ，且最相近的 feature < 第二相近的 feature * 0.8，做 matching

### (4) Image Matching and Stitching
我們使用 RANSAC 演算法決定拼接時兩張照片的平移量。
**RANSAC Algorithm**

**Why not using homography matrix?**
由於我們發現 Homography matrix 會對照片產生 translation, 而拼接到越後面的照片，累積的 translation 將越嚴重，導致後面的照片嚴重扭曲。故我們選擇計算兩張照片的平移量，來代替 homography matrix.

### (5) Blending


## 5. Result
TODO

## 6. Summary

我們完成了以下work:
- 實作
- 實作 
- 實作

## 7. Reproduce Steps
TODO


