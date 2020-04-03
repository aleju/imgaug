# imgaug

이 파이썬 라이브러리는 기계 학습 프로젝트를 위한 이미지를 보강하는데 도움이 된다.

Imgaug는 입력된 이미지 세트를 훨씬 더 큰 약간 변경된 새로운 이미지 세트로 변환한다.

[![Build Status](https://travis-ci.org/aleju/imgaug.svg?branch=master)](https://travis-ci.org/aleju/imgaug)
[![codecov](https://codecov.io/gh/aleju/imgaug/branch/master/graph/badge.svg)](https://codecov.io/gh/aleju/imgaug)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1370ce38e99e40af842d47a8dd721444)](https://www.codacy.com/app/aleju/imgaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aleju/imgaug&amp;utm_campaign=Badge_Grade)

<table>

<tr>
<th>&nbsp;</th>
<th>이미지</th>
<th>히트맵</th>
<th>분할 지도</th>
<th>주요지점</th>
<th>경계 상자,<br>다각형</th>
</tr>

<!-- Line 1: Original Input(입력값) -->
<tr>
<td><em>입력값</em></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_image.jpg?raw=true" height="83" width="124" alt="input images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_heatmap.jpg?raw=true" height="83" width="124" alt="input heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_segmap.jpg?raw=true" height="83" width="124" alt="input segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_kps.jpg?raw=true" height="83" width="124" alt="input keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_bbs.jpg?raw=true" height="83" width="124" alt="input bounding boxes"></td>
</tr>

<!-- Line 2: 가우스. 잡읍 + 대조 + 갈다 -->
<tr>
<td>가우스. 잡음<br>+&nbsp;대조<br>+&nbsp;갈다</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_image.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_heatmap.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_segmap.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_kps.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_bbs.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 3: Affine(벡터화) -->
<tr>
<td>벡터화</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_image.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_heatmap.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_segmap.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_kps.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_bbs.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 4: Crop + Pad (다듬기+채우기) -->
<tr>
<td>다듬기<br>+&nbsp;채우기</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_image.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_heatmap.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_segmap.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_kps.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_bbs.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 5: Fliplr + Perspective -->
<tr>
<td>배열 뒤집기<br>+&nbsp;원근법</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_image.jpg" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_heatmap.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_segmap.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_kps.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_bbs.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to bounding boxes"></td>
</tr>

</table>


**하나의 입력 이미지의 더 많은 (강력한) 예제 기능:**

![64 quokkas](https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/examples_grid.jpg?raw=true "64 quokkas")


## 목록

1. [특징](#feature)
2. [설치](#installation)
3. [참고자료](#documentation)
4. [최근 변경사항](#recent_changes)
5. [예시 이미지](#example_images)
6. [코드 예제](#code_examples)
7. [증강 목록](#list_of_augmenters)
8. [인용](#citation)


<a name="feature"/>

## 특징

* 많은 증강 기법
  * 예.아핀 변환, 원근법 변환, 대조 변화, 가우시안 소음, 지역 중퇴, 색상 / 포화 변화, 자르기 / 패딩, 흐림, ...
* 고성능 최적화
  * 일부 이미지에만 확대를 적용하기 쉽다.
  * 무작위로 증강을 적용하기 쉽다
* 다음을 지원한다.
  * 이미지 (uint8에 대한 전체 지원, 다른 데이터타입에 대한 지원은 [문서](https://imgaug.readthedocs.io/en/latest/source/dtype_support.html)를 참조하십시오.)
  * 히트맵 (float32), 분할 맵 (int), Masks (bool)
    * 해당 이미지보다 작거나 클 수 있다. 예.자르기 같은 기능을 위한 추가 코드 라인이 필요 *없다*. 
  * 키포인트/랜드마크 (int/float coordinates)
  * 경계상자 (int/float coordinates)
  * 다각형 (int/float coordinates) (Beta)
  * 선 문자열 (int/float coordinates) (Beta)
* 샘플링된 랜덤 값의 자동 정렬
  * 예: 이미지와 분할 맵을 균일한 값(-10°, 45°)으로 샘플링하여 동일 값으로 회전시킨다. (0개의 추가 코드 라인)
* 확률 분포를 매개변수로 사용
  * 예: 이미지들을 균일(-10°, 45°)에서 샘플링한 값들로 회전시킨다.
  * 예: `ABS(.)`는 절대함수, `N(.)`는 가우시안 분포, 그리고 `B(.)`는 베타 분포일때, `ABS(N(0, 20.0))*(1+B(1.0, 1.0))`에서 샘플링된 값으로 이미지들을 회전시킨다.
* 많은 도우미 기능
  * 예 : 열 지도, 분할 지도, 키포인트, 경계 상자등 그리기, ...
  * 예 : 축척 분할 맵, 이미지 / 맵의 평균 / 최대 풀, 패드 이미지 대 종횡비 (예. 제곱)
  * 예 : 키포인트를 거리 맵으로 변환, 바운딩 상자 내의 픽셀을 이미지에서 추출, 폴리곤을 이미지 평면으로 클립, ...
* 다수의 CPU 코어에서 증강 지원


<a name="installation"/>

## 설치

이 라이브러리는 파이썬 2.7과 3.4+를 지원한다.

### 설치: 아나콘다

아나콘다에 라이브러리를 설치하려면 다음 명령을 수행하십시오:
```bash
conda config --add channels conda-forge
conda install imgaug
```

`conda remove imgaug`를 통해 라이브러리를 삭제할 수 있다.

### 설치: pip

pip을 통해 라이브러리를 설치하려면 먼저 모든 요구 사항을 설치하십시오:
```bash
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
```

그런 다음, pypi를 통해 imgaug를 설치하십시오 (github 버전에 따라 지연될 수 있다):
```bash
pip install imgaug
```

또는 github에서 최신 버전을 직접 설치하십시오:
```bash
pip install git+https://github.com/aleju/imgaug.git
```

드물게, `Shapely` 설치 문제를 일으킬 수 있다.
이 경우 패키지를 건너뛸 수 있다 -- 하지만 폴리곤과 선 문자열 증강은
`Shapely`없이는 충돌할 수 있다.

라이브러리를 삭제하기 위해서는, `pip uninstall imgaug`를 실행하면 된다.

### 설치: From Source

또는, 저장소를 통해 다운로드할 수 있다.
`git clone https://github.com/aleju/imgaug` 그리고 다음 명령어를 통해 수동으로 설치할 수 있다.
`cd imgaug && python setup.py install`.


<a name="documentation"/>

## 참고자료

jupyter notebooks 예시:
  * [이미지를 로드하고 증강한다](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb)
  * [다중 증강](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb)
  * 증강 및 작업 : [키포인트/랜드마크](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb),
    [경계 상자](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb),
    [다각형](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B03%20-%20Augment%20Polygons.ipynb),
    [선 문자열](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B06%20-%20Augment%20Line%20Strings.ipynb),
    [히트맵](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B04%20-%20Augment%20Heatmaps.ipynb),
    [분할맵](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B05%20-%20Augment%20Segmentation%20Maps.ipynb) 

더 많은 자료: [imgaug-doc/notebooks](https://github.com/aleju/imgaug-doc/tree/master/notebooks).

ReadTheDocs 페이지 예제 (보통 notebooks보다 최신 버전이 적다):
* [라이브러리 사용 방법에 대한 간단한 예제 코드](http://imgaug.readthedocs.io/en/latest/source/examples_basics.html)
* [지원되는 증강 기술 예제](http://imgaug.readthedocs.io/en/latest/source/augmenters.html)
* [API](http://imgaug.readthedocs.io/en/latest/source/api.html)

더 많은 RTD 자료: [imgaug.readthedocs.io](http://imgaug.readthedocs.io/en/latest/source/examples_basics.html).

이 프로젝트의 모든 문서 관련 파일은 다음 저장소에 호스팅되어 있다. [imgaug-doc](https://github.com/aleju/imgaug-doc).


<a name="recent_changes"/>

## 최근 변경사항

* **0.3.0**: 재 작업된 분할 맵 증강이 numpy 1.17+에 적용되었다.
  난수 샘플링 API, 몇 개의 새로운 증강기.
* **0.2.9**: 추가된 폴리곤 증강, 추가된 선 문자열 증강,
단순화된 증강 인터페이스.
* **0.2.8**: 향상된 성능, dtype 지원 및 멀티 코어 증강.

자세한 내용은 [changelogs/](changelogs/)를 참조하십시오.


<a name="example_images"/>

## 예시 이미지

아래 이미지는 대부분의 증강 기법에 대한 예제를 보여준다.

(a, b) 형태로 쓰여진 값은 균일한 분포를 나타내며,
즉, 값은 간격 `[a, b]`에서 랜덤하게 선택된다.
선 문자열은 모든 증강자에서 지원되지만 여기서 명시적으로 시각화되지는 않는다.

<table>

<tr><td colspan="5"><strong>meta</strong></td></tr>
<tr>
<td colspan="1"><sub>Noop</sub></td>
<td colspan="1"><sub>ChannelShuffle</sub></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/noop.gif" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/channelshuffle.gif" height="148" width="100" alt="ChannelShuffle"></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>arithmetic</strong></td></tr>
<tr>
<td colspan="1"><sub>Add</sub></td>
<td colspan="1"><sub>Add<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>AdditiveGaussianNoise</sub></td>
<td colspan="1"><sub>AdditiveGaussianNoise<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>AdditiveLaplaceNoise</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/add.gif" height="148" width="100" alt="Add"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/add_per_channel_true.gif" height="148" width="100" alt="Add per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivegaussiannoise.gif" height="148" width="100" alt="AdditiveGaussianNoise"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivegaussiannoise_per_channel_true.gif" height="148" width="100" alt="AdditiveGaussianNoise per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivelaplacenoise.gif" height="148" width="100" alt="AdditiveLaplaceNoise"></td>
</tr>
<tr>
<td colspan="1"><sub>AdditiveLaplaceNoise<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>AdditivePoissonNoise</sub></td>
<td colspan="1"><sub>AdditivePoissonNoise<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>Multiply</sub></td>
<td colspan="1"><sub>Multiply<br/>(per_channel=True)</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivelaplacenoise_per_channel_true.gif" height="148" width="100" alt="AdditiveLaplaceNoise per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivepoissonnoise.gif" height="148" width="100" alt="AdditivePoissonNoise"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/additivepoissonnoise_per_channel_true.gif" height="148" width="100" alt="AdditivePoissonNoise per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/multiply.gif" height="148" width="100" alt="Multiply"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/multiply_per_channel_true.gif" height="148" width="100" alt="Multiply per_channel=True"></td>
</tr>
<tr>
<td colspan="1"><sub>Dropout</sub></td>
<td colspan="1"><sub>Dropout<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>CoarseDropout<br/>(p=0.2)</sub></td>
<td colspan="1"><sub>CoarseDropout<br/>(p=0.2, per_channel=True)</sub></td>
<td colspan="1"><sub>ImpulseNoise</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/dropout.gif" height="148" width="100" alt="Dropout"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/dropout_per_channel_true.gif" height="148" width="100" alt="Dropout per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/coarsedropout_p_0_2.gif" height="148" width="100" alt="CoarseDropout p=0.2"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/coarsedropout_p_0_2_per_channel_true.gif" height="148" width="100" alt="CoarseDropout p=0.2, per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/impulsenoise.gif" height="148" width="100" alt="ImpulseNoise"></td>
</tr>
<tr>
<td colspan="1"><sub>SaltAndPepper</sub></td>
<td colspan="1"><sub>Salt</sub></td>
<td colspan="1"><sub>Pepper</sub></td>
<td colspan="1"><sub>CoarseSaltAndPepper<br/>(p=0.2)</sub></td>
<td colspan="1"><sub>CoarseSalt<br/>(p=0.2)</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/saltandpepper.gif" height="148" width="100" alt="SaltAndPepper"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/salt.gif" height="148" width="100" alt="Salt"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/pepper.gif" height="148" width="100" alt="Pepper"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/coarsesaltandpepper_p_0_2.gif" height="148" width="100" alt="CoarseSaltAndPepper p=0.2"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/coarsesalt_p_0_2.gif" height="148" width="100" alt="CoarseSalt p=0.2"></td>
</tr>
<tr>
<td colspan="1"><sub>CoarsePepper<br/>(p=0.2)</sub></td>
<td colspan="1"><sub>Invert</sub></td>
<td colspan="1"><sub>Invert<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>JpegCompression</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/coarsepepper_p_0_2.gif" height="148" width="100" alt="CoarsePepper p=0.2"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/invert.gif" height="148" width="100" alt="Invert"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/invert_per_channel_true.gif" height="148" width="100" alt="Invert per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/jpegcompression.gif" height="148" width="100" alt="JpegCompression"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>blend</strong></td></tr>
<tr>
<td colspan="1"><sub>Alpha<br/>with EdgeDetect(1.0)</sub></td>
<td colspan="1"><sub>Alpha<br/>with EdgeDetect(1.0)<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>SimplexNoiseAlpha<br/>with EdgeDetect(1.0)</sub></td>
<td colspan="1"><sub>FrequencyNoiseAlpha<br/>with EdgeDetect(1.0)</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/alpha_with_edgedetect_1_0.gif" height="148" width="100" alt="Alpha with EdgeDetect1.0"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/alpha_with_edgedetect_1_0_per_channel_true.gif" height="148" width="100" alt="Alpha with EdgeDetect1.0 per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/simplexnoisealpha_with_edgedetect_1_0.gif" height="148" width="100" alt="SimplexNoiseAlpha with EdgeDetect1.0"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/frequencynoisealpha_with_edgedetect_1_0.gif" height="148" width="100" alt="FrequencyNoiseAlpha with EdgeDetect1.0"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>blur</strong></td></tr>
<tr>
<td colspan="1"><sub>GaussianBlur</sub></td>
<td colspan="1"><sub>AverageBlur</sub></td>
<td colspan="1"><sub>MedianBlur</sub></td>
<td colspan="1"><sub>BilateralBlur<br/>(sigma_color=250,<br/>sigma_space=250)</sub></td>
<td colspan="1"><sub>MotionBlur<br/>(angle=0)</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/gaussianblur.gif" height="148" width="100" alt="GaussianBlur"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/averageblur.gif" height="148" width="100" alt="AverageBlur"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/medianblur.gif" height="148" width="100" alt="MedianBlur"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/bilateralblur_sigma_color_250_sigma_space_250.gif" height="148" width="100" alt="BilateralBlur sigma_color=250, sigma_space=250"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/motionblur_angle_0.gif" height="148" width="100" alt="MotionBlur angle=0"></td>
</tr>
<tr>
<td colspan="1"><sub>MotionBlur<br/>(k=5)</sub></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/motionblur_k_5.gif" height="148" width="100" alt="MotionBlur k=5"></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>color</strong></td></tr>
<tr>
<td colspan="1"><sub>MultiplyHueAndSaturation</sub></td>
<td colspan="1"><sub>MultiplyHue</sub></td>
<td colspan="1"><sub>MultiplySaturation</sub></td>
<td colspan="1"><sub>AddToHueAndSaturation</sub></td>
<td colspan="1"><sub>AddToHue</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/multiplyhueandsaturation.gif" height="148" width="100" alt="MultiplyHueAndSaturation"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/multiplyhue.gif" height="148" width="100" alt="MultiplyHue"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/multiplysaturation.gif" height="148" width="100" alt="MultiplySaturation"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/addtohueandsaturation.gif" height="148" width="100" alt="AddToHueAndSaturation"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/addtohue.gif" height="148" width="100" alt="AddToHue"></td>
</tr>
<tr>
<td colspan="1"><sub>AddToSaturation</sub></td>
<td colspan="1"><sub>Grayscale</sub></td>
<td colspan="1"><sub>KMeansColorQuantization<br/>(to_colorspace=RGB)</sub></td>
<td colspan="1"><sub>UniformColorQuantization<br/>(to_colorspace=RGB)</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/addtosaturation.gif" height="148" width="100" alt="AddToSaturation"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/grayscale.gif" height="148" width="100" alt="Grayscale"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/kmeanscolorquantization_to_colorspace_rgb.gif" height="148" width="100" alt="KMeansColorQuantization to_colorspace=RGB"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/uniformcolorquantization_to_colorspace_rgb.gif" height="148" width="100" alt="UniformColorQuantization to_colorspace=RGB"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>contrast</strong></td></tr>
<tr>
<td colspan="1"><sub>GammaContrast</sub></td>
<td colspan="1"><sub>GammaContrast<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>SigmoidContrast<br/>(cutoff=0.5)</sub></td>
<td colspan="1"><sub>SigmoidContrast<br/>(gain=10)</sub></td>
<td colspan="1"><sub>SigmoidContrast<br/>(per_channel=True)</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/gammacontrast.gif" height="148" width="100" alt="GammaContrast"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/gammacontrast_per_channel_true.gif" height="148" width="100" alt="GammaContrast per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/sigmoidcontrast_cutoff_0_5.gif" height="148" width="100" alt="SigmoidContrast cutoff=0.5"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/sigmoidcontrast_gain_10.gif" height="148" width="100" alt="SigmoidContrast gain=10"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/sigmoidcontrast_per_channel_true.gif" height="148" width="100" alt="SigmoidContrast per_channel=True"></td>
</tr>
<tr>
<td colspan="1"><sub>LogContrast</sub></td>
<td colspan="1"><sub>LogContrast<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>LinearContrast</sub></td>
<td colspan="1"><sub>LinearContrast<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>AllChannels-<br/>HistogramEqualization</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/logcontrast.gif" height="148" width="100" alt="LogContrast"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/logcontrast_per_channel_true.gif" height="148" width="100" alt="LogContrast per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/linearcontrast.gif" height="148" width="100" alt="LinearContrast"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/linearcontrast_per_channel_true.gif" height="148" width="100" alt="LinearContrast per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/allchannels_histogramequalization.gif" height="148" width="100" alt="AllChannels- HistogramEqualization"></td>
</tr>
<tr>
<td colspan="1"><sub>HistogramEqualization</sub></td>
<td colspan="1"><sub>AllChannelsCLAHE</sub></td>
<td colspan="1"><sub>AllChannelsCLAHE<br/>(per_channel=True)</sub></td>
<td colspan="1"><sub>CLAHE</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/histogramequalization.gif" height="148" width="100" alt="HistogramEqualization"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/allchannelsclahe.gif" height="148" width="100" alt="AllChannelsCLAHE"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/allchannelsclahe_per_channel_true.gif" height="148" width="100" alt="AllChannelsCLAHE per_channel=True"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/clahe.gif" height="148" width="100" alt="CLAHE"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>convolutional</strong></td></tr>
<tr>
<td colspan="1"><sub>Sharpen<br/>(alpha=1)</sub></td>
<td colspan="1"><sub>Emboss<br/>(alpha=1)</sub></td>
<td colspan="1"><sub>EdgeDetect</sub></td>
<td colspan="1"><sub>DirectedEdgeDetect<br/>(alpha=1)</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/sharpen_alpha_1.gif" height="148" width="100" alt="Sharpen alpha=1"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/emboss_alpha_1.gif" height="148" width="100" alt="Emboss alpha=1"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/edgedetect.gif" height="148" width="100" alt="EdgeDetect"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/directededgedetect_alpha_1.gif" height="148" width="100" alt="DirectedEdgeDetect alpha=1"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>edges</strong></td></tr>
<tr>
<td colspan="1"><sub>Canny</sub></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/canny.gif" height="148" width="100" alt="Canny"></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>flip</strong></td></tr>
<tr>
<td colspan="2"><sub>Fliplr</sub></td>
<td colspan="2"><sub>Flipud</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/fliplr.gif" height="148" width="300" alt="Fliplr"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/flipud.gif" height="148" width="300" alt="Flipud"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>geometric</strong></td></tr>
<tr>
<td colspan="2"><sub>Affine</sub></td>
<td colspan="2"><sub>Affine: Modes</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/affine.gif" height="148" width="300" alt="Affine"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/affine_modes.gif" height="148" width="300" alt="Affine: Modes"></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><sub>Affine: cval</sub></td>
<td colspan="2"><sub>PiecewiseAffine</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/affine_cval.gif" height="148" width="300" alt="Affine: cval"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/piecewiseaffine.gif" height="148" width="300" alt="PiecewiseAffine"></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><sub>PerspectiveTransform</sub></td>
<td colspan="2"><sub>ElasticTransformation<br/>(sigma=0.2)</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/perspectivetransform.gif" height="148" width="300" alt="PerspectiveTransform"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/elastictransformation_sigma_0_2.gif" height="148" width="300" alt="ElasticTransformation sigma=0.2"></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><sub>ElasticTransformation<br/>(sigma=5.0)</sub></td>
<td colspan="2"><sub>Rot90</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/elastictransformation_sigma_5_0.gif" height="148" width="300" alt="ElasticTransformation sigma=5.0"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/rot90.gif" height="148" width="300" alt="Rot90"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>pooling</strong></td></tr>
<tr>
<td colspan="1"><sub>AveragePooling</sub></td>
<td colspan="1"><sub>MaxPooling</sub></td>
<td colspan="1"><sub>MinPooling</sub></td>
<td colspan="1"><sub>MedianPooling</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/averagepooling.gif" height="148" width="100" alt="AveragePooling"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/maxpooling.gif" height="148" width="100" alt="MaxPooling"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/minpooling.gif" height="148" width="100" alt="MinPooling"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/medianpooling.gif" height="148" width="100" alt="MedianPooling"></td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>segmentation</strong></td></tr>
<tr>
<td colspan="1"><sub>Superpixels<br/>(p_replace=1)</sub></td>
<td colspan="1"><sub>Superpixels<br/>(n_segments=100)</sub></td>
<td colspan="1"><sub>UniformVoronoi</sub></td>
<td colspan="1"><sub>RegularGridVoronoi: rows/cols<br/>(p_drop_points=0)</sub></td>
<td colspan="1"><sub>RegularGridVoronoi: p_drop_points<br/>(n_rows=n_cols=30)</sub></td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/superpixels_p_replace_1.gif" height="148" width="100" alt="Superpixels p_replace=1"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/superpixels_n_segments_100.gif" height="148" width="100" alt="Superpixels n_segments=100"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/uniformvoronoi.gif" height="148" width="100" alt="UniformVoronoi"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/regulargridvoronoi_rows_cols_p_drop_points_0.gif" height="148" width="100" alt="RegularGridVoronoi: rows/cols p_drop_points=0"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/regulargridvoronoi_p_drop_points_n_rows_n_cols_30.gif" height="148" width="100" alt="RegularGridVoronoi: p_drop_points n_rows=n_cols=30"></td>
</tr>
<tr>
<td colspan="1"><sub>RegularGridVoronoi: p_replace<br/>(n_rows=n_cols=16)</sub></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/regulargridvoronoi_p_replace_n_rows_n_cols_16.gif" height="148" width="100" alt="RegularGridVoronoi: p_replace n_rows=n_cols=16"></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>size</strong></td></tr>
<tr>
<td colspan="2"><sub>CropAndPad</sub></td>
<td colspan="2"><sub>Crop</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/cropandpad.gif" height="148" width="300" alt="CropAndPad"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/crop.gif" height="148" width="300" alt="Crop"></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><sub>Pad</sub></td>
<td colspan="2"><sub>PadToFixedSize<br/>(height'=height+32,<br/>width'=width+32)</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/pad.gif" height="148" width="300" alt="Pad"></td>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/padtofixedsize_height_height_32_width_width_32.gif" height="148" width="300" alt="PadToFixedSize height'=height+32, width'=width+32"></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><sub>CropToFixedSize<br/>(height'=height-32,<br/>width'=width-32)</sub></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/croptofixedsize_height_height_32_width_width_32.gif" height="148" width="300" alt="CropToFixedSize height'=height-32, width'=width-32"></td>
<td>&nbsp;</td>
<td>&nbsp;</td>
<td>&nbsp;</td>
</tr>
<tr><td colspan="5"><strong>weather</strong></td></tr>
<tr>
<td colspan="1"><sub>FastSnowyLandscape<br/>(lightness_multiplier=2.0)</sub></td>
<td colspan="1"><sub>Clouds</sub></td>
<td colspan="1"><sub>Fog</sub></td>
<td colspan="1"><sub>Snowflakes</sub></td>
<td>&nbsp;</td>
</tr>
<tr>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/fastsnowylandscape_lightness_multiplier_2_0.gif" height="144" width="128" alt="FastSnowyLandscape lightness_multiplier=2.0"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/clouds.gif" height="144" width="128" alt="Clouds"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/fog.gif" height="144" width="128" alt="Fog"></td>
<td colspan="1"><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/augmenter_videos/snowflakes.gif" height="144" width="128" alt="Snowflakes"></td>
<td>&nbsp;</td>
</tr>

</table>



<a name="code_examples"/>


## 코드 예제

### 예시: 간단한 학습 배치

표준 기계 학습 상황.
이미지 배치 교육 및 크롭, 수평 플립("Fliplr") 및 gaussian blur를 통해 각 배치 확대:
```python
import numpy as np
import imgaug.augmenters as iaa

def load_batch(batch_idx):
    # dummy function, implement this
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image
    # sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.
    return np.zeros((128, 32, 32, 3), dtype=np.uint8) + (batch_idx % 255)

def train_on_images(images):
    # dummy function, implement this
    pass

# Pipeline:
# (1) Crop images from each side by 1-16px, do not resize the results
#     images back to the input size. Keep them at the cropped size.
# (2) Horizontally flip 50% of the images.
# (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
seq = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

for batch_idx in range(100):
    images = load_batch(batch_idx)
    images_aug = seq(images=images)  # done by the library
    train_on_images(images_aug)
```


### 예제: 매우 복잡한 확장 파이프라인

이미지에 매우 무거운 증강 pipeline 적용(이 판독값 맨 위에 이미지를 생성하는 데 사용):
```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)
images_aug = seq(images=images)
```


### 예시: 확장 이미지와 핵심요소

동일한 영상의 영상 및 핵심요소/랜드마크 확대:
```python
import numpy as np
import imgaug.augmenters as iaa

images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
points = [
    [(10.5, 20.5)],  # points on first image
    [(50.5, 50.5), (60.5, 60.5), (70.5, 70.5)]  # points on second image
]

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Affine(translate_px={"x": (1, 5)})
])

# augment keypoints and images
images_aug, points_aug = seq(images=images, keypoints=points)

print("Image 1 center", np.argmax(images_aug[0, 64, 64:64+6, 0]))
print("Image 2 center", np.argmax(images_aug[1, 64, 64:64+6, 0]))
print("Points 1", points_aug[0])
print("Points 2", points_aug[1])
```
`imgaug`의 모든 좌표는 서브픽셀 정확도로 `x=0.5, y=0.5`는 왼쪽 상단 픽셀의 픽셀 중심을 나타낸다.

### 예시: 확장 이미지 및 경계 상자

```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
bbs = [
    [ia.BoundingBox(x1=10.5, y1=15.5, x2=30.5, y2=50.5)],
    [ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=50.5),
     ia.BoundingBox(x1=40.5, y1=75.5, x2=70.5, y2=100.5)]
]

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Affine(translate_px={"x": (1, 5)})
])

images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
```


### 예시: 증강 이미지 및 다각형

```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
polygons = [
    [ia.Polygon([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
    [ia.Polygon([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0)])]
]

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Affine(translate_px={"x": (1, 5)})
])

images_aug, polygons_aug = seq(images=images, polygons=polygons)
```


### 예시: 확장 이미지 및 LineStrings

LineStrings는 폴리곤과 유사하지만 닫히지 않으며, 자신과 교차하며 내부 영역이 없을 수 있다.
```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
ls = [
    [ia.LineString([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
    [ia.LineString([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0),
                    (128.0, 0.0)])]
]

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Affine(translate_px={"x": (1, 5)})
])

images_aug, ls_aug = seq(images=images, line_strings=ls)
```


### 예시: 확장 이미지 및 히트맵

히트맵은 `0.0`~`1.0`의 값을 갖는 고밀도 플로트 배열이다.
예를 들어 얼굴 특징 위치를 예측하기 위해 모델을 훈련할 때 사용할 수 있다.
히트맵은 이미지보다 낮은 높이와 폭을 가지고 있다는 점에 유의해야한다.
`imgaug`는 그 일을 자동으로 처리한다.
크롭 픽셀 양은 히트맵의 경우 반으로 줄어들 것이다.

```python
import numpy as np
import imgaug.augmenters as iaa

# Standard scenario: You have N RGB-images and additionally 21 heatmaps per
# image. You want to augment each image and its heatmaps identically.
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
heatmaps = np.random.random(size=(16, 64, 64, 1)).astype(np.float32)

seq = iaa.Sequential([
    iaa.GaussianBlur((0, 3.0)),
    iaa.Affine(translate_px={"x": (-40, 40)}),
    iaa.Crop(px=(0, 10))
])

images_aug, heatmaps_aug = seq(images=images, heatmaps=heatmaps)
```


### 예시: 확장 이미지 및 세그먼트 맵

이는 히트맵과 비슷하지만 촘촘한 배열은 dtype `int32`을 가지고 있다.
크기 조정과 같은 작업은 자동으로 가장 가까운 인접 보조를 사용한다.

```python
import numpy as np
import imgaug.augmenters as iaa

# Standard scenario: You have N=16 RGB-images and additionally one segmentation
# map per image. You want to augment each image and its heatmaps identically.
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
segmaps = np.random.randint(0, 10, size=(16, 64, 64, 1), dtype=np.int32)

seq = iaa.Sequential([
    iaa.GaussianBlur((0, 3.0)),
    iaa.Affine(translate_px={"x": (-40, 40)}),
    iaa.Crop(px=(0, 10))
])

images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)
```


### 예시: 증강된 이미지 시각화

확대된 시퀀스의 예제 결과를 표시:
```python
import numpy as np
import imgaug.augmenters as iaa

images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

# Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
# versions of image 1. Identical augmentations will be applied to
# image 0 and 1.
seq.show_grid([images[0], images[1]], cols=8, rows=8)
```


### 예시: 증강된 이미지가 아닌 데이터 시각화하기

imgaug에는 바운딩 박스나 히트맵과 같은 이미지가 아닌 결과를 빠르게 시각화 할수 있는 많은 기능이 포함되어 있다.

```python
import numpy as np
import imgaug as ia

image = np.zeros((64, 64, 3), dtype=np.uint8)

# points
kps = [ia.Keypoint(x=10.5, y=20.5), ia.Keypoint(x=60.5, y=60.5)]
kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
image_with_kps = kpsoi.draw_on_image(image, size=7, color=(0, 0, 255))
ia.imshow(image_with_kps)

# bbs
bbsoi = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=30.5)
], shape=image.shape)
image_with_bbs = bbsoi.draw_on_image(image)
image_with_bbs = ia.BoundingBox(
    x1=50.5, y1=10.5, x2=100.5, y2=16.5
).draw_on_image(image_with_bbs, color=(255, 0, 0), size=3)
ia.imshow(image_with_bbs)

# polygons
psoi = ia.PolygonsOnImage([
    ia.Polygon([(10.5, 20.5), (50.5, 30.5), (10.5, 50.5)])
], shape=image.shape)
image_with_polys = psoi.draw_on_image(
    image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0))
ia.imshow(image_with_polys)

# heatmaps
hms = ia.HeatmapsOnImage(np.random.random(size=(32, 32, 1)).astype(np.float32),
                         shape=image.shape)
image_with_hms = hms.draw_on_image(image)
ia.imshow(image_with_hms)
```

LineStrings 과 segmentation maps 도 위와 같은 방법을 지원한다. 


### 예시: 증강 한 번만 사용하기 

인터페이스는 기능 보강 인스턴스를 여러 번 재사용하도록 조정되어 있지만 한 번만 자유롭게 사용할 수도 있다. augmenter를 매번 인스턴스화하는 오버 헤드는 대개 무시할 만하다.

```python
from imgaug import augmenters as iaa
import numpy as np

images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# always horizontally flip each input image
images_aug = iaa.Fliplr(1.0)(images=images)

# vertically flip each input image with 90% probability
images_aug = iaa.Flipud(0.9)(images=images)

# blur 50% of all images using a gaussian kernel with a sigma of 3.0
images_aug = iaa.Sometimes(0.5, iaa.GaussianBlur(3.0))(images=images)
```


### 예시: 멀티코어 증강

이미지는 `augment_batches(batches, background=True)`방식을 이용하여 백그라운드 프로세스에서 보강될 수 있다. `batches`는
[imgaug.augmentables.batches.UnnormalizedBatch](https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmentables/batches.html#UnnormalizedBatch)
or
[imgaug.augmentables.batches.Batch](https://imgaug.readthedocs.io/en/latest/source/api_augmentables_batches.html#imgaug.augmentables.batches.Batch).
의 목록/생성기이다.
아래의 예는 백그라운드에서 이미지 batch를 보강한다.
```python
import skimage.data
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch

# Number of batches and batch size for this example
nb_batches = 10
batch_size = 32

# Example augmentation sequence to run in the background
augseq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CoarseDropout(p=0.1, size_percent=0.1)
])

# For simplicity, we use the same image here many times
astronaut = skimage.data.astronaut()
astronaut = ia.imresize_single_image(astronaut, (64, 64))

# Make batches out of the example image (here: 10 batches, each 32 times
# the example image)
batches = []
for _ in range(nb_batches):
    batches.append(UnnormalizedBatch(images=[astronaut] * batch_size))

# Show the augmented images.
# Note that augment_batches() returns a generator.
for images_aug in augseq.augment_batches(batches, background=True):
    ia.imshow(ia.draw_grid(images_aug.images_aug, cols=8))
```

백그라운드 augmentation에 더 많은 통제가 필요하다면, (예: 시드 설정, 사용 된 CPU 코어 수 제어 또는 메모리 사용량 제한)
그에 해당하는
[multicore augmentation notebook](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb)
이나
[Augmenter.pool()](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Augmenter.pool)
과
[imgaug.multicore.Pool](https://imgaug.readthedocs.io/en/latest/source/api_multicore.html#imgaug.multicore.Pool).
에 대한 API 를 참조하시오.

### 예시: 매개변수로서의 확률 분포

대부분의 augmenter는 튜플  `(a, b)`을 `uniform(a, b)`을 나타내는 바로 가기로 사용하거나 목록 `[a, b, c]`를 사용하여 하나를 임의로 선택할 수있는 허용 된 값 세트를 나타낸다. 더 복잡한 확률 분포 (예 : 가우시안, 잘린 가우시안 또는 포아송 분포)가 필요한 경우 `imgaug.parameters`에서 확률 매개 변수를 사용할 수 있다.

```python
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap

images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Blur by a value sigma which is sampled from a uniform distribution
# of range 10.1 <= x < 13.0.
# The convenience shortcut for this is: GaussianBlur((10.1, 13.0))
blurer = iaa.GaussianBlur(10 + iap.Uniform(0.1, 3.0))
images_aug = blurer(images=images)

# Blur by a value sigma which is sampled from a gaussian distribution
# N(1.0, 0.1), i.e. sample a value that is usually around 1.0.
# Clip the resulting value so that it never gets below 0.1 or above 3.0.
blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(1.0, 0.1), 0.1, 3.0))
images_aug = blurer(images=images)
```

라이브러리에는 더 많은 확률 분포가 있다. (예 : 절단 된 가우시안 분포, 포아송 분포 또는 베타 분포.) 


### 예시: WithChannels

특정 이미지 채널에만 증강을 적용: 
```python
import numpy as np
import imgaug.augmenters as iaa

# fake RGB images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# add a random value from the range (-30, 30) to the first two channels of
# input images (e.g. to the R and G channels)
aug = iaa.WithChannels(
  channels=[0, 1],
  children=iaa.Add((-30, 30))
)

images_aug = aug(images=images)
```


### 예시: Hooks

미리 정해진 순서에 따라 증강을 자유롭게 비활성화 할 수 있다. 여기서는 파이프 라인을 통해 두 번째 배열(`heatmaps`)을 실행하여, 해당 입력에 증강의 subset만 적용한다. 
```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# Images and heatmaps, just arrays filled with value 30.
# We define the heatmaps here as uint8 arrays as we are going to feed them
# through the pipeline similar to normal images. In that way, every
# augmenter is applied to them.
images = np.full((16, 128, 128, 3), 30, dtype=np.uint8)
heatmaps = np.full((16, 128, 128, 21), 30, dtype=np.uint8)

# add vertical lines to see the effect of flip
images[:, 16:128-16, 120:124, :] = 120
heatmaps[:, 16:128-16, 120:124, :] = 120

seq = iaa.Sequential([
  iaa.Fliplr(0.5, name="Flipper"),
  iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
  iaa.Dropout(0.02, name="Dropout"),
  iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise"),
  iaa.AdditiveGaussianNoise(loc=32, scale=0.0001*255, name="SomeOtherNoise"),
  iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
])

# change the activated augmenters for heatmaps,
# we only want to execute horizontal flip, affine transformation and one of
# the gaussian noises
def activator_heatmaps(images, augmenter, parents, default):
    if augmenter.name in ["GaussianBlur", "Dropout", "MyLittleNoise"]:
        return False
    else:
        # default value for all other augmenters
        return default
hooks_heatmaps = ia.HooksImages(activator=activator_heatmaps)

# call to_deterministic() once per batch, NOT only once at the start
seq_det = seq.to_deterministic()
images_aug = seq_det(images=images)
heatmaps_aug = seq_det(images=heatmaps, hooks=hooks_heatmaps)
```


<a name="list_of_augmenters"/>

## 증강 목록

다음은 사용 가능한 증강의 목록이다. 아래에 언급 된 대부분의 변수는 범위로 설정할 수 있다. (예 : 이미지 당 0과 1.0 사이의 임의의 값을 샘플링하려면 `A=(0.0, 1.0)`, 이미지 당 `0.0`이나 `0.5` 또는 `1.0`을 임의로 샘플링하려면 `A=[0.0, 0.5, 1.0]`. 

**산수**

| 증강 | 정의 |
| --- | --- |
| Add(V, PCH) | `V`값을 각 이미지에 추가한다. `PCH`가 참이라면, 샘플 값이 채널마다 달라진다.  |
| AddElementwise(V, PCH) | `V`값을 각 픽셀 단위에 추가한다. `PCH`가 참이라면, 샘플 값이 채널마다 달라진다. (픽셀 마다)  |
| AdditiveGaussianNoise(L, S, PCH) | 픽셀단위의 화이트 노이즈와 가우시안 노이즈를 이미지에 첨가한다. 노이즈는 정규 분포 `N(L,S)` 를 따른다. `PCH`가 참이라면, 샘플 값이 채널마다 달라진다. (픽셀 마다) |
| AdditiveLaplaceNoise(L, S, PCH) | `Laplace(L, S)`에 따라 laplace 분포에서 샘플링 된 노이즈를 이미지에 추가한다. `PCH`가 참이라면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있다. |
| AdditivePoissonNoise(L, PCH) | L이 람다 지수 인 포아송 분포에서 샘플링 된 노이즈를 추가한다. `PCH`가 참이라면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있다. |
| Multiply(V, PCH) | 각 이미지에 `V`값을 곱하여 더 어둡고 밝은 이미지로 만든다. `PCH`가 참이면 샘플링 된 값이 채널마다 다를 수 있다. |
| MultiplyElementwise(V, PCH) | 각 픽셀에 값 `V`를 곱하여 더 어둡고 밝은 픽셀로 만든다. `PCH`가 참이면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있다.  |
| Dropout(P, PCH) |확률이 P 인 픽셀을 0으로 설정한다. `PCH`가 참이면 채널이 다르게 처리 될 수 있으며, 그렇지 않으면 전체 픽셀이 0으로 설정된다.  |
| CoarseDropout(P, SPX, SPC, PCH) | `Dropout`과 유사하지만 픽셀 크기가 `SPX`이거나 상대적 크기가 `SPC` 인 거친 / 작은 이미지에서 0으로 설정 될 픽셀의 위치를 샘플링한다. 즉 `SPC`에 작은 값이 있으면 대략적인 맵이 작으므로 큰 사각형이 삭제된다. |
| ReplaceElementwise(M, R, PCH) | 이미지의 픽셀을 `R`로 대체한다. 마스크 `M`으로 식별된 픽셀로 대체한다. `M`은 확률이 될 수 있다. 예를 들어 모든 픽셀의 5 %를 대체하려면 `0.05`이다. `PCH`가 참이면 마스크는 이미지, 픽셀 및 추가로 채널별로 샘플링된다.|
| ImpulseNoise(P) | 모든 픽셀의 `P` 퍼센트를 임펄스 노이즈, 즉 매우 밝거나 어두운 RGB 색상으로 대체한다. `SaltAndPepper(P, PCH=True)`와 같다.  |
| SaltAndPepper(P, PCH) | 모든 픽셀의 `P` 퍼센트를 매우 흰색 또는 검은 색으로 바꾼다. PCH가 참이면 채널마다 다른 픽셀이 교체된다. |
| CoarseSaltAndPepper(P, SPX, SPC, PCH) |`CoarseDropout`과 유사하지만 영역을 0으로 설정하는 대신 매우 흰색 또는 검은 색으로 바꾼다. `PCH`가 참이면, coarse 교체 마스크는 이미지 및 채널당 한 번 샘플링된다.  |
| Salt(P, PCH) | `SaltAndPepper`와 유사하지만 검은 색이 아닌 매우 흰색으로만 대체된다.  |
| CoarseSalt(P, SPX, SPC, PCH) | `CoarseSaltAndPepper`와 유사하지만 검은 색이 아닌 매우 흰색으로만 대체된다. |
| Pepper(P, PCH) | `SaltAndPepper`와 유사하지만 흰 색이 아닌 매우 검은 색으로만 대체된다.  |
| CoarsePepper(P, SPX, SPC, PCH) | `CoarseSaltAndPepper`와 유사하지만 흰 색이 아닌 매우 검은 색으로만 대체된다. |
| Invert(P, PCH) | 이미지의 모든 픽셀을 확률 P로 반전한다. 즉, (1-pixel_value)로 설정한다. PCH가 참이면 각 채널이 개별적으로 처리된다 (일부 채널 만 반전 됨).  |
| ContrastNormalization(S, PCH) | 픽셀 값을 128보다 가까이 또는 더 가깝게 이동하여 이미지의 차이를 변경한다. 방향과 강도는 S로 정의된다. PCH가 true로 설정되면 프로세스는 다른 가능한 S로 채널 단위로 발생한다.  |
| JpegCompression(C) | 강도 C (값 범위 : 0 ~ 100)의 JPEG 압축을 이미지에 적용한다. C 값이 높을수록 시각적 인공물이 더 많이 나타난다.  |


**혼합**

| 증강 | 정의 |
| --- | --- |
| Alpha(A, FG, BG, PCH) | 증강 `FG`와 `BG`를 사용하여 이미지를 보강 한 다음 alpha `A`를 사용하여 결과를 혼합한다. FG와 BG는 기본적으로 제공되지 않으면 아무 것도 수행하지 않는다. 예 : `Alpha(0.9, FG)`를 사용하여 `FG`를 통해 이미지를 확대 한 다음 결과를 혼합하여 원래 이미지의 10 %를 유지한다 (`FG` 이전). PCH가 true로 설정되면 프로세스는 A와 다르게 채널 단위로 발생한다 (FG 및 BG는 이미지 당 한 번 계산 됨).  |
| AlphaElementwise(A, FG, BG, PCH) |`Alpha`와 동일하지만 A에서 샘플링 된 연속 마스크 (값 0.0 ~ 1.0)를 사용하여 픽셀 단위로 블렌딩을 수행한다. PCH가 true로 설정되면 프로세스는 픽셀 단위와 채널 단위로 발생한다.  |
| SimplexNoiseAlpha(FG, BG, PCH, SM, UP, I, AGG, SIG, SIGT) | `Alpha`와 유사하지만 마스크를 사용하여 증강 FG 및 BG의 결과를 혼합한다. 마스크는 단순 노이즈에서 샘플링되며, 이는 거친 경향이 있다. 마스크는 I 반복 (기본값 : 1 ~ 3)으로 수집되며 각 반복은 집계 방법 AGG (기본 최대, 즉 픽셀 당 모든 반복의 최대 값)를 사용하여 결합된다. 각 마스크는 최대 해상도 SM (기본값 2 ~ 16px)의 저해상도 공간에서 샘플링되며 UP 방법 (기본값 : 선형 또는 3 차 또는 가장 가까운 인접 업 샘플링)을 사용하여 이미지 크기로 업 스케일된다. SIG가 true이면 임계 값 SIGT를 사용하여 S 자형이 마스크에 적용되어 블롭의 값이 0.0 또는 1.0에 가까워진다. |
| FrequencyNoiseAlpha(E, FG, BG, PCH, SM, UP, I, AGG, SIG, SIGT) |`SimplexNoiseAlpha`와 유사하지만 주파수 영역에서 노이즈 마스크를 생성한다. 지수 E는 주파수 성분을 증가 / 감소시키는 데 사용된다. E의 값이 높으면 고주파 성분이 발음된다. 대략 -2에서 생성 된 구름 같은 패턴과 함께 -4에서 4 사이의 값을 사용하시오.  |


**블러**

| 증강 | 정의 |
| --- | --- |
| GaussianBlur(S) | 크기가 S 인 가우스 커널을 사용하여 이미지를 흐리게한다. |
| AverageBlur(K) | 크기가 K 인 간단한 averaging 커널을 사용하여 이미지를 흐리게한다. |
| MedianBlur(K) | K 크기의 중간 값을 통해 중앙값을 사용하여 이미지를 흐리게한다. |
| BilateralBlur(D, SC, SS) | 거리 D (커널 크기 등)의 양방향 필터를 사용하여 이미지를 흐리게한다. SC는 색 공간의 (영향) 거리에 대한 시그마이고, SS는 공간 거리에 대한 시그마이다.  |
| MotionBlur(K, A, D, O) | 크기가 K 인 모션 블러 커널을 사용하여 이미지를 흐리게한다. A는 y 축에 대한 흐림 각도이다 (값 범위 : 0-360, 시계 방향). D는 흐림 방향이다 (값 범위 : -1.0 ~ 1.0, 1.0은 중앙에서 앞으로). O은 보간 순서이다 (O = 0은 빠름, O = 1은 약간 느리지 만 더 정확하다). |


**색상**

| 증강 | 정의 |
| --- | --- |
| WithColorspace(T, F, CH) | 색상 공간 T에서 F로 이미지를 변환하고 자식 증강 CH를 적용한 다음 F에서 T로 다시 변환한다. |
| AddToHueAndSaturation(V, PCH, F, C) | HSV 공간의 각 픽셀에 값 V를 추가한다 (예 : 색조 및 채도 수정). 색 공간 F에서 HSV로 변환한다 (기본값은 F = RGB). 증강 하기 전에 채널 C를 선택한다 (기본값은 C = [0,1]). PCH가 참이면 샘플링 된 값이 채널마다 다를 수 있다.  |
| ChangeColorspace(T, F, A) | 색상 공간 F에서 T로 이미지를 변환하고 alpha 'A'를 사용하여 원본 이미지와 혼합한다. 회색조는 3 채널로 유지된다. (실제로 테스트되지 않은 증강이므로 위험이 감수 될수 있다.)  |
| Grayscale(A, F) | 색상 공간 F (기본값 : RGB)에서 이미지를 회색조로 변환하고 alpha 'A'를 사용하여 원본 이미지와 혼합한다.  |


**대조**

| 증강 | 정의 |
| --- | --- |
| GammaContrast(G, PCH) | `I_ij' = I_ij**G'`다음에 감마 대비 조정을 적용한다. 여기서 `G'`는 G에서 샘플링 된 감마 값이고 픽셀에서 `I_ij` (0에서 1.0 공간으로 변환)이다. PCH가 참이면 이미지와 채널마다 다른 `G'`가 샘플링된다. |
| SigmoidContrast(G, C, PCH) | GammaContrast와 유사하지만 `I_ij' = 1/(1 + exp(G' * (C' - I_ij)))`를 적용한다. 여기서 `G'`는 G에서 샘플링 된 이득 값이고`C'`는 C에서 샘플링 된 손실 값이다.  |
| LogContrast(G, PCH) | GammaContrast와 유사하지만 `I_ij = G' * log(1 + I_ij)`를 적용한다. 여기서 `G'`는 G에서 샘플링 된 이득 값이다.  |
| LinearContrast(S, PCH) | GammaContrast와 유사하지만 `I_ij = 128 + S' * (I_ij - 128)`를 적용한다. 여기서 `S'`는 S에서 샘플링 된 강도 값이다. 이 증강은 ContrastNormalization과 동일하다 (향후 더 이상 사용되지 않음). |
| AllChannelsHistogramEqualization() | 각 입력 이미지의 각 채널에 표준 Histogram Equalization을 적용한다.  |
| HistogramEqualization(F, T) | `AllChannelsHistogramEqualization` 와 유사하지만 이미지가 색상 공간 F에있을 것으로 예상하고 색상 공간 T로 변환하고 강도 관련 채널 만 정규화한다 (예 : T = Lab의 경우 L (T의 기본값) 또는 T = HSV의 V이다. | 
| AllChannelsCLAHE(CL, K, Kmin, PCH) | 클리핑 제한 CL 및 커널 사이즈 K (범위 `[Kmin, inf)`로 클리핑 됨)를 사용하여 각 이미지 채널에 적용되는 Limited Adaptive Histrogram Equalization을 대조한다.(작은 이미지 패치의 Histogram Equalization). PCH가 참이면 채널마다 CL 및 K에 대한 다른 값이 샘플링된다. |
| CLAHE(CL, K, Kmin, F, T) | `HistogramEqualization`과 유사하게 Lab / HSV / HLS 색 공간의 강도 관련 채널에만 CLAHE를 적용한다. (일반적으로 이것은 `AllChannelsCLAHE`보다 훨씬 잘 작동한다.)  |


**합성**

| 증강 | 정의 |
| --- | --- |
| Convolve(M) | 람다 함수일 가능성이 있는 행렬 M으로 이미지를 통합한다.  |
| Sharpen(A, L) | 밝기 L로 각 이미지에 선명하게 커널을 실행한다 (값이 낮으면 이미지가 어두워진다). Alpha 'A'를 사용하여 결과를 원본 이미지와 혼합한다. |
| Emboss(A, S) | 강도가 S 인 각 이미지에서 emboss 커널을 실행한다. Alpha 'A'를 사용하여 결과를 원본 이미지와 혼합한다. |
| EdgeDetect(A) | 각 이미지에서 edge detection 커널을 실행한다. Alpha 'A'를 사용하여 결과를 원본 이미지와 혼합한다. |
| DirectedEdgeDetect(A, D) | 각 이미지에 대해 방향 지정 edge detection 커널을 실행하여 각 방향 D에서 감지한다 (기본값 : 이미지 당 선택한 0에서 360 도의 임의 방향). Alpha 'A'를 사용하여 결과를 원본 이미지와 혼합한다. |


**엣지**

| 증강 | 정의 |
| --- | --- |
| Canny(A, HT, SK, C) | 히스테리시스 임계 값 HT 및 소벨 커널 크기 SK를 사용하여 각 이미지에 canny edge detection을 적용한다. 클래스 C를 사용하여 이진 이미지를 색상으로 변환한다. Alpha는 요소 A를 사용하여 입력 이미지와 혼합한다. |


**뒤집기**

| 증강 | 정의 |
| --- | --- |
| Fliplr(P) | 확률 P로 이미지를 가로로 뒤집는다. |
| Flipud(P) | 확률 P로 이미지를 세로로 뒤집는다. |


**기하**

| 증강 | 정의 |
| --- | --- |
| Affine(S, TPX, TPC, R, SH, O, CVAL, FO, M, B) | 이미지에 아핀 변환을 적용한다. S로 스케일을 조정하고 (> 1 = 확대, <1 = 확대), TPX 픽셀 또는 TPC 백분율로 변환하고, R 도씩 회전하고 SH도만큼 기울인다. 순서 O로 보간이 발생한다 (0 또는 1이 양호하고 빠름). FO가 참이면 출력 이미지 평면 크기가 왜곡 된 이미지 크기에 맞춰진다. 즉 45도 회전 한 이미지는 이미지 평면 외부에 있지 않다. M은 입력 이미지 평면에 해당하지 않는 출력 이미지 평면의 픽셀을 처리하는 방법을 제어한다. `M='constant'`이면 CVAL은 이러한 픽셀을 채울 상수 값을 정의한다. B는 백엔드 프레임 워크 (현재 `cv2` 또는 `skimage`)를 설정할 수 있다. |
| AffineCv2(S, TPX, TPC, R, SH, O, CVAL, M, B) | 'Affine'과 동일하지만 백엔드로 cv2 만 사용한다. 현재 FO = true를 지원하지 않는다. 향후에는 더 이상 사용되지 않을 수 있다. |
| PiecewiseAffine(S, R, C, O, M, CVAL) |이미지에 일정한 점 격자를 배치한다. 그리드에는 R 행과 C 열이 있다. 그런 다음 정규 분포 N (0, S)의 샘플 인 양만큼 점 (및 그 주변의 이미지 영역)을 이동하여 다양한 강도의 local distortion을 일으킨다. O, M 및 CVAL은 `Affine`에서와 같이 정의된다.  |
| PerspectiveTransform(S, KS) |임의의 4 점 투시 변환을 이미지에 적용한다 (advanced 클리핑 형태와 유사). 각 점은 시그마 S를 사용한 정규 분포에서 파생 된 이미지의 코너로부터 임의의 거리를 갖는다. KS가 True (기본값)로 설정되면 각 이미지의 크기가 원래 크기로 다시 조정된다. |
| ElasticTransformation(S, SM, O, CVAL, M) | 왜곡 필드를 기준으로 각 픽셀을 개별적으로 이동한다. SM은 왜곡 필드의 평활도와 S 강도를 정의한다. O는 보간 순서이며, CVAL은 새로 생성 된 픽셀에 대한 상수 채우기 값이고 M은 채우기 모드이다 (증강 `Affine` 참조). |
| Rot90(K, KS) | 이미지를 시계 방향으로 90도 회전한다. (이것은 `Affine`보다 빠르다.) KS가 true이면 결과 이미지는 원래 입력 이미지와 동일한 크기로 크기가 조정된다. |


**메타**

| 증강 | 정의 |
| --- | --- |
| Sequential(C, R) |자식 증강 `C`의 목록을 가져 와서 이 순서대로 이미지에 적용한다. R이 true이면 (기본값 : false) 순서는 무작위이다 (배치 당 한 번 선택). |
| SomeOf(N, C, R) | 증강 `C` 목록에서 임의로 선택된 N 개의 증강을 각 이미지에 적용한다. 증강은 이미지마다 선택된다. R은 `Sequential`과 동일하다. N은 범위 일 수 있다 (예 : 1에서 3을 선택하기 위해 (1, 3). |
| OneOf(C) | SomeOf(1, C)와 동일. |
| Sometimes(P, C, D) | 자식 증강 `C`를 사용하여 확률 P로 이미지를 보강하고, 그렇지 않으면 D를 사용한다. D는 없음 일 수 있으며, 모든 이미지의 P % 만 `C`를 통해 증강.  |
| WithColorspace(T, F, C) | 이미지를 색상 공간 F (기본값 : RGB)에서 색상 공간 T로 변환하고, 증강 C를 적용한 다음 다시 F로 변환한다.  |
| WithChannels(H, C) | 각 이미지 채널 H (예 : RGB 이미지에서 빨강 및 녹색의 경우 `[0,1]`)에서 선택하고 자식 증강 `C`를 이 채널에 적용하고 결과를 원래 이미지로 다시 병합한다.  |
| Noop() | 아무것도 하지 않는다. (검증 / 테스트에 유용하다.) |
| Lambda(I, K) | 람다 함수 I을 이미지에 적용하고 K를 keypoint에 적용한다. |
| AssertLambda(I, K) | 람다 함수 I을 통해 이미지를 확인하고 K를 통해 keypoint를 확인하고 둘 중 하나에 의해 false가 반환되면 오류가 발생한다. |
| AssertShape(S) | 입력 이미지의 모양이 'S'가 아닌 경우 오류가 발생한다.  |
| ChannelShuffle(P, C) | 모든 이미지의 P 퍼센트에 대한 색상 채널 순서를 변경한다. 기본적으로 모든 채널을 셔플하지만 'C' (채널 인덱스 목록)를 사용하는 부분집합으로 제한 할 수 있다.  |


**풀링**

| 증강 | 정의 |
| --- | --- |
| AveragePooling(K, KS) |커널 크기가 K 인 평균 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하시오.  |
| MaxPooling(K, KS) | 커널 크기가 K 인 최대 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하시오.  |
| MinPooling(K, KS) | 커널 크기가 K 인 최소 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하시오.  |
| MedianPooling(K, KS) | 커널 크기가 K 인 중앙 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하시오. |


**분할**

| 증강 | 정의 |
| --- | --- |
| Superpixels(P, N, M) | (최대) 해상도 M에서 이미지의 N 수퍼 픽셀을 생성하고 원래 크기로 다시 크기를 조정한다. 그런 다음 원본 이미지의 모든 수퍼 픽셀 영역의 P %가 수퍼 픽셀로 대체된다. (1-P) 퍼센트는 변경되지 않는다.  |
| Voronoi(PS, P, M) |Voronoi 셀의 좌표를 얻기 위해 샘플러 PS를 쿼리한다. 각 셀에서 모든 픽셀을 프로브로 바꾼다. 평균으로 P. 최대 해상도 M에서 이 단계를 수행한다. |
| UniformVoronoi(N, P, M) | 각 이미지에 `N` Voronoi 셀을 무작위로 배치한다. 각 셀에서 모든 픽셀을 프로브로 바꾼다. 평균으로 P. 최대 해상도 M에서이 단계를 수행한다. |
| RegularGridVoronoi(H, W, P, M) |각 이미지에 'H'x'W' (높이 x 너비) Voronoi 셀의 규칙적인 그리드를 배치한다. 각 셀에서 모든 픽셀을 프로브로 바꾼다. 평균으로 P. 최대 해상도 M에서이 단계를 수행한다. |
| RelativeRegularGridVoronoi(HPC, WPC, P, M) |각 이미지에 `HPC*H x WPC*W` Voronoi 셀의 규칙적인 그리드를 배치한다 (H, W는 이미지 높이 및 너비). 각 셀에서 모든 픽셀을 프로브로 바꾼다. 평균으로 P. 최대 해상도 M에서이 단계를 수행한다. |


**크기**

| 증강 | 정의 |
| --- | --- |
| Resize(S, I) | 이미지의 크기를 S로 조정한다. 일반적인 사용 사례는 `S={"height":H, "width":W}`를 사용하여 모든 이미지의 크기를 `HxW` 모양으로 조정하는 것이다. H와 W는 플로트 일 수 있다 (예 : 원래 크기의 50 %로 크기 조정). H 또는 W는 한쪽의 새 크기만 정의하고 다른 쪽의 크기를 적절하게 조정하기 위해 `"종횡비 유지"`일 수 있다. I는 (기본값 : `cubic`)을 사용하기 위한 보간이다. |
| CropAndPad(PX, PC, PM, PCV, KS) | 이미지의 위 / 오른쪽 / 아래 / 왼쪽에서 PX 픽셀 또는 픽셀의 PC 백분율을 자르거나 채운다. 음수 값은 잘라내기, 양수 값은 채우기를 실행한다. PM은 패드 모드를 정의한다 (예 : 추가 된 모든 픽셀에 균일 한 색상 사용). PMV가 일정한 경우 PCV는 추가 된 픽셀의 색상을 제어한다. KS가 true (기본값)이면 결과 이미지가 원래 크기로 다시 조정된다. |
| Pad(PX, PC, PM, PCV, KS) | CropAndPad () 의 픽셀을 추가하기만 하는 숏컷이다. PX 및 PC에는 양수 값만 허용된다.  |
| Crop(PX, PC, KS) | CropAndPad ()의 픽셀을 잘라내기만 하는 숏컷이다. PX 및 PC에는 양수 값만 사용할 수 있다 (예 : 5 값은 5 픽셀이 잘린다).  |
| PadToFixedSize(W, H, PM, PCV, POS) |높이 H와 너비 W까지의 모든 이미지를 채운다. PM 및 PCV는 'Pad'와 동일하다. POS는 패딩 할 위치를 정의한다. POS = "center"는 모든면에 똑같이, POS = "left-top"은 윗면과 왼쪽만 채운다.  |
| CropToFixedSize(W, H, POS) | `PadToFixedSize`와 비슷하지만 패딩 대신 높이 H와 너비 W로 자른다.  |
| KeepSizeByResize(CH, I, IH) | 자식 증강 CH (예 : 자르기)를 적용한 후 모든 이미지의 크기를 원래 크기로 다시 조정한다. I는 이미지에 사용 된 보간이고, IH는 히트맵에 사용되는 보간이다.  |


**날씨**

| 증강 | 정의 |
| --- | --- |
| FastSnowyLandscape(LT, LM) | HLS 색상 공간에서 `L<LT`를 갖는 모든 픽셀의 밝기 L을 LM의 계수로 증가시켜 풍경 이미지를 눈 풍경으로 변환  |
| Clouds() | 다양한 모양과 밀도의 구름을 이미지에 추가한다. *오버레이* 증강과 같이 쓰는 것이 좋다.(예:`SimplexNoiseAlpha`) |
| Fog() | 다양한 모양과 밀도의 안개 같은 구름 구조를 이미지에 추가한다. *오버레이* 증강과 같이 쓰는 것이 좋다. (예:`SimplexNoiseAlpha`)  |
| CloudLayer(IM, IFE, ICS, AMIN, AMUL, ASPXM, AFE, S, DMUL) | 이미지에 단일 구름 레이어를 추가한다. IM은 구름의 평균 강도, IFE는 강도에 대한 주파수 노이즈 지수 (고르지 않은 색상으로 이어짐), ICS는 강도 샘플링을 위한 가우시안의 분산을 제어하고 AM은 구름의 최소 불투명도 (값> 0은 일반적인 안개), 불투명도 값의 승수 AMUL, ASPXM은 불투명도 값을 샘플링 할 최소 그리드 크기를 제어하고, AFE는 불투명도 값의 주파수 노이즈 지수, S는 구름의 희소성을 제어하고 DMUL은 구름 밀도 멀티 플라이어이다. 이 인터페이스는 최종적이 아니며 향후 변경 될 수 있다.  |
| Snowflakes(D, DU, FS, FSU, A, S) | 밀도 D, 밀도 균일도 DU, 눈송이 크기 FS, 눈송이 크기 균일도 FSU, 떨어지는 각도 A 및 속도 S를 가진 눈송이를 이미지에 추가한다. 1 ~ 3 층의 눈송이가 추가되므로 값은 확률론적이어야 한다.  |
| SnowflakesLayer(D, DU, FS, FSU, A, S, BSF, BSL) | 이미지에 눈송이의 단일 레이어를 추가한다. `Snowflakes` 증강을 참조하시오. BSF 및 BSL은 눈송이에 적용되는 가우시안 블러를 제어한다. |


<a name="citation"/>

## 인용

이 라이브러리가 연구에 도움이 되었다면, 마음대로 인용해도 좋음 :
```latex
@misc{imgaug,
  author = {Jung, Alexander B.
            and Wada, Kentaro
            and Crall, Jon
            and Tanaka, Satoshi
            and Graving, Jake
            and Yadav, Sarthak
            and Banerjee, Joy
            and Vecsei, Gábor
            and Kraft, Adam
            and Borovec, Jirka
            and Vallentin, Christian
            and Zhydenko, Semen
            and Pfeiffer, Kilian
            and Cook, Ben
            and Fernández, Ismael
            and Weng Chi-Hung
            and Ayala-Acevedo, Abner
            and Meudec, Raphael
            and Laporte, Matias
            and others},
  title = {{imgaug}},
  howpublished = {\url{https://github.com/aleju/imgaug}},
  year = {2019},
  note = {Online; accessed 25-Sept-2019}
}
```

