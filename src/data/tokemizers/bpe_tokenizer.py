class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)
        pass

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора
        pass

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # TODO: Реализуйте метод декодирования предсказанных токенов
        pass
