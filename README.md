# VFX_hw2
R11946012 資科所 王奕方 
R10942198 電信丙 林仲偉

## 1. Description
在定點拍攝全景場景不同方向的照片，先使用 feature detection 找出兩兩照片之間的特徵點，然後使用 feature matching 找出 match point pairs，最後根據 match point pairs 做 image stitching，並使用 blendering 使得拼接後影像邊界視覺上看起來沒有接縫，以得到一張全景圖。

## 2. Experiment Setup

| 項目 | 描述                  |
|:---- |:--------------------- |
| 機型 |              |
| 鏡頭 |             |
| 焦距 |                  |
| ISO  |                    |
| F    |                    |


### 場景圖片
| 項目   | 描述      |
|:------ |:--------- |
| 尺寸   |   4000x6000 |
| 解析度 |  350 |
| 數量   | 20        |


* 場景ㄧ：水源會館前院

<img src="https://i.imgur.com/YcTHH4F.jpg" width="200px"><img src="https://i.imgur.com/2cdx7Mb.jpg" width="200px"><img src="https://i.imgur.com/A4tJUP6.jpg" width="200px"><img src="https://i.imgur.com/XuGh8AX.jpg" width="200px"><img src="https://i.imgur.com/C9L4TYF.jpg" width="200px"><img src="https://i.imgur.com/ctXEbej.jpg" width="200px"><img src="https://i.imgur.com/PZ9fVN7.jpg" width="200px"><img src="https://i.imgur.com/chZOxN4.jpg" width="200px"><img src="https://i.imgur.com/LZq1xG9.jpg" width="200px"><img src="https://i.imgur.com/aziLJpl.jpg" width="200px"><img src="https://i.imgur.com/u5AV2A5.jpg" width="200px"><img src="https://i.imgur.com/3J5ZA9Q.jpg" width="200px"><img src="https://i.imgur.com/36IXExP.jpg" width="200px"><img src="https://i.imgur.com/XvdCesB.jpg" width="200px"><img src="https://i.imgur.com/RRlrtxM.jpg" width="200px"><img src="https://i.imgur.com/WnqBNdF.jpg" width="200px"><img src="https://i.imgur.com/GmeeQIe.jpg" width="200px"><img src="https://i.imgur.com/ZcvRQaD.jpg" width="200px"><img src="https://i.imgur.com/8RrGOfI.jpg" width="200px"><img src="https://i.imgur.com/YYy9uNb.jpg" width="200px">


* 場景二：水源會館草皮

<img src="https://i.imgur.com/YK1o1TQ.jpg" width="200px"><img src="https://i.imgur.com/K9GCiaA.jpg" width="200px"><img src="https://i.imgur.com/skdS66n.jpg" width="200px"><img src="https://i.imgur.com/T6AY1w6.jpg" width="200px"><img src="https://i.imgur.com/Rnbpj6Z.jpg" width="200px"><img src="https://i.imgur.com/L2oDk5N.jpg" width="200px"><img src="https://i.imgur.com/WpaD11s.jpg" width="200px"><img src="https://i.imgur.com/1JVlEP7.jpg" width="200px"><img src="https://i.imgur.com/kt6c68g.jpg" width="200px"><img src="https://i.imgur.com/WvIXFFd.jpg" width="200px"><img src="https://i.imgur.com/FKl4cB4.jpg" width="200px"><img src="https://i.imgur.com/B5jDUQU.jpg" width="200px"><img src="https://i.imgur.com/nnYkWP7.jpg" width="200px"><img src="https://i.imgur.com/QymFXLs.jpg" width="200px"><img src="https://i.imgur.com/1FK3vjl.jpg" width="200px"><img src="https://i.imgur.com/mzczKYr.jpg" width="200px"><img src="https://i.imgur.com/TYSzyMv.jpg" width="200px"><img src="https://i.imgur.com/NrgTIZR.jpg" width="200px"><img src="https://i.imgur.com/yDWNKhO.jpg" width="200px"><img src="https://i.imgur.com/GkWtlST.jpg" width="200px">


## 3. Program Workflow
1. 使用 autostitch 去得到所有照片的 focal length
2. 對照片做 cylindrical projection
3. 使用 Harris Corner Detector/SIFT 去做 Feature detection
4. Feature matching
5. 用 RANSAC 去找出使得 Image matching 結果最好的 Homography matrix (在此作業中，我們假設只會發生平移)
6. 根據 Homography matrix 對兩張圖片做 Image Stitching.
7. 做 Linear Blending
8. 重複 2~7, 直到所有照片都被拼接完成。 


## 4. Implementation Detail

### (1) Cylindrical Projection

### (2) Feature Detection: 

**Harris Corner Detector:**

**SIFT:**

### (3) Feature Matching

### (4) Image Matching

**RANSAC Algorithm:**

### (5) Image Stitching

### (6) Blending


## 5. Result
TODO

## 6. Summary

我們完成了以下work:
- 實作
- 實作 
- 實作

## 7. Reproduce Steps
TODO


