import torch

def merge_subwords_and_scores(tokens, scores):
    """
    특수 토큰([CLS], [SEP], [PAD])을 제외하고, 
    ## 접두어가 붙은 토큰들을 이전 단어에 병합하며 점수를 합산합니다.
    """
    merged_words = []
    merged_scores = []
    
    # TODO: 토큰 리스트와 스코어 리스트를 받아 아래 규칙대로 병합을 구현하세요.
    # 1. 특수 토큰([CLS], [SEP], [PAD]) 제외
    # 2. ##으로 시작하는 토큰은 이전 단어에 이어 붙이고 점수를 합산(또는 평균)
    # 3. 그 외 토큰은 새 단어로 추가
    
    return merged_words, merged_scores

def explain_integrated_gradients(model, tokenizer, text, max_length=128, device="cpu", steps=50):
    """
    텍스트에 대해 Integrated Gradients 중요도를 계산하고 어절 단위로 병합된 중요도 점수를 반환합니다.
    
    Args:
        model: Hugging Face Sequence Classification 모델
        tokenizer: AutoTokenizer 객체
        text: 설명하려는 단일 문장 (str)
        max_length: 입력 시퀀스 최대 길이
        device: 계산에 사용할 PyTorch 디바이스
        steps: 리만 합(Riemann sum) 적분을 위한 보간 스텝 수 (기본값: 50)
    
    Returns:
        dict: {
            "words": 복원된 단어(어절) 리스트,
            "scores": 각 단어별 Integrated Gradients 기여도 점수,
            "prediction": 모델의 예측 클래스 (0: 부정, 1: 긍정),
            "probability": 예측된 클래스의 확률
        }
    """
    # TODO: Integrated Gradients 알고리즘을 구현하세요.
    # 1. 모델을 eval 모드로 설정
    # 2. 입력 텍스트 토큰화 및 디바이스 이동
    # 3. 모델의 Embedding layer(model.get_input_embeddings())를 통해 입력 임베딩(inputs_embeds) 획득
    # 4. baseline 임베딩(예: zero embedding) 설정 및 steps 수만큼 선형 보간 임베딩 생성 (requires_grad_() 설정 필요)
    # 5. 원래 모델의 예측 타겟 클래스(예측 확률이 가장 높은 클래스) 결정
    # 6. 보간된 임베딩들을 모델에 통과시키고, 타겟 클래스 logit에 대한 임베딩의 기울기(gradient) 계산
    # 7. 기울기들의 평균(Riemann Sum) 계산 후 (input_embeds - baseline_embeds)와 곱하여 토큰별 기여도 구함
    # 8. 임베딩 차원(hidden_dim)에 대해 합산(sum)하여 각 토큰별 1차원 점수 획득
    # 9. merge_subwords_and_scores 함수를 호출하여 어절 단위로 복원된 결과 반환
    
    # 임시 반환값 예시 (구현 후 제거하세요)
    return {
        "words": ["예시", "단어"],
        "scores": [0.0, 0.0],
        "prediction": 0,
        "probability": 0.5
    }

if __name__ == "__main__":
    print("explain_ig.py 스켈레톤 로드가 완료되었습니다.")
