use tokenizers::Tokenizer;

pub struct StreamDecoder;

impl StreamDecoder {
    pub fn decode_next(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let prev_text = tokenizer.decode(&tokens[..tokens.len() - 1], true).unwrap_or_default();
        let new_text = tokenizer.decode(tokens, true).unwrap_or_default();

        if new_text.len() > prev_text.len() {
            let split_idx = prev_text.len();
            if new_text.is_char_boundary(split_idx) {
                return new_text[split_idx..].to_string();
            }
        }

        String::new()
    }
}