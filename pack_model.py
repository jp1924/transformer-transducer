from models import (
    TransformerTransducerConfig,
    TransformerTransducerFeatureExtractor,
    TransformerTransducerForRNNT,
    TransformerTransducerProcessor,
    TransfoXLConfig,
)

from transformers import AddedToken, AutoConfig, AutoModel, AutoTokenizer


BLK_TOKEN = "<blank>"


def main() -> None:
    text_model_name = ""
    save_path = ""

    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    tokenizer.add_tokens(AddedToken(BLK_TOKEN, special=True, normalized=False), special_tokens=True)

    new_vocab_size = len(tokenizer.get_vocab())
    text_model = AutoModel.from_pretrained(text_model_name)
    new_embeding = text_model.resize_token_embeddings(new_vocab_size)
    text_model.set_input_embeddings(new_embeding)

    text_config = AutoConfig.from_pretrained(
        text_model_name,
        vocab_size=new_vocab_size,
        padding_idx=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )
    audio_config = TransfoXLConfig()
    config = TransformerTransducerConfig(
        text_config=text_config.to_dict(),
        audio_config=audio_config.to_dict(),
        vocab_size=new_vocab_size,
        blk_token_ids=tokenizer.convert_tokens_to_ids(BLK_TOKEN),
    )
    model = TransformerTransducerForRNNT(config)
    model.text_model = text_model
    feature_extractor = TransformerTransducerFeatureExtractor()
    processor = TransformerTransducerProcessor(feature_extractor, tokenizer)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    TransformerTransducerForRNNT.from_pretrained(save_path)
    TransformerTransducerProcessor.from_pretrained(save_path)


if "__main__" in __name__:
    main()
