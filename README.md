# Fashion-MNIST-CNN

Fashion-MNIST 의류 이미지 분류 프로젝트

## Project Structure

### Scratch

- 밑바닥부터 시작하는 딥러닝 소스를 활용한 CNN 구현
- NumPy를 이용하여 Convolution, Pooling, Affine 계층 직접 사용
- Adam Optimizer, Dropout, 하이퍼파라미터 튜닝 적용

### PyTorch

- PyTorch를 이용한 CNN 구현
- Fashion-MNIST 데이터셋 학습 및 평가

## Dataset

- Fashion-MNIST
- Train: 60,000 images
- Test: 10,000 images
- Image Size: 28×28 grayscale

## Hyperparameters (Scratch)

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 100
- Epochs:20
- Filter Number:50
- Hidden Size:200
- Dropout:0.3

## Results

### Scratch

- Train Accuracy:97.7%
- Test Accuracy:91.84%

### PyTorch

- Train Accuracy: 96.68%
- Test Accuracy: 94.04%
