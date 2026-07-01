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

def explain_occlusion(model, tokenizer, text, max_length=128, device="cpu"):
    """
    텍스트의 각 토큰을 하나씩 [PAD]로 가린 뒤, 예측 확률의 변화량(P_orig - P_new)을 기반으로 중요도를 산출합니다.
    
    Args:
        model: Hugging Face Sequence Classification 모델
        tokenizer: AutoTokenizer 객체
        text: 설명하려는 단일 문장 (str)
        max_length: 입력 시퀀스 최대 길이
        device: 계산에 사용할 PyTorch 디바이스
    
    Returns:
        dict: {
            "words": 복원된 단어(어절) 리스트,
            "scores": 각 단어별 Occlusion 중요도 점수,
            "prediction": 모델의 예측 클래스 (0: 부정, 1: 긍정),
            "probability": 예측된 클래스의 확률
        }
    """
    # TODO: Occlusion 알고리즘을 구현하세요.
    # 1. 모델을 eval 모드로 설정
    # 2. 입력 텍스트 토큰화 및 디바이스 이동
    # 3. 원본 입력 문장의 예측 결과(예측 클래스 및 원래 예측 확률 P_orig) 획득
    # 4. 각 토큰 위치 i를 순회하며:
    #    - 특수 토큰([CLS], [SEP], [PAD])인 경우 점수 0 부여 후 건너뜀
    #    - i번째 토큰의 ID를 pad_token_id로 바꾼 변형된 입력을 모델에 통과시킴
    #    - 변형된 입력에서 타겟 클래스의 새로운 예측 확률 P_new 획득
    #    - 중요도 점수 = P_orig - P_new 계산 후 저장
    # 5. merge_subwords_and_scores 함수를 호출하여 어절 단위로 복원된 결과 반환
    
    # 임시 반환값 예시 (구현 후 제거하세요)
    return {
        "words": ["예시", "단어"],
        "scores": [0.0, 0.0],
        "prediction": 0,
        "probability": 0.5
    }

if __name__ == "__main__":
    print("explain_occlusion.py 스켈레톤 로드가 완료되었습니다.")
