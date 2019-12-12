import tensorflow as tf
import tensorflow_datasets as tfds

import re

def _re_compile(pattern):
  return re.compile(pattern, flags=re.UNICODE)

NUM_BYTES = 2**8
ALPHANUM_REGEX = _re_compile(r"\W+")
ALL_REGEX = _re_compile(r"(\W+)")

class CustomTokenTextEncoder(tfds.features.text.TextEncoder):
  r"""TextEncoder backed by a list of tokens.
  Tokenization splits on (and drops) non-alphanumeric characters with
  regex "\W+".
  """

  def __init__(self,
               vocab_list,
               oov_buckets=1,
               oov_token="UNK",
               lowercase=False,
               strip_vocab=True,
               decode_token_separator=" "):
    """Constructs a TokenTextEncoder.
    To load from a file saved with `TokenTextEncoder.save_to_file`, use
    `TokenTextEncoder.load_from_file`.
    Args:
      vocab_list: `list<str>`, list of tokens.
      oov_buckets: `int`, the number of `int`s to reserve for OOV hash buckets.
        Tokens that are OOV will be hash-modded into a OOV bucket in `encode`.
      oov_token: `str`, the string to use for OOV ids in `decode`.
      lowercase: `bool`, whether to make all text and tokens lowercase.
        list of tokens.
      strip_vocab: `bool`, whether to strip whitespace from the beginning and
        end of elements of `vocab_list`.
      decode_token_separator: `str`, the string used to separate tokens when
        decoding.
    """
    self._vocab_list = [tf.compat.as_text(el) for el in vocab_list]
    if strip_vocab:
      self._vocab_list = [el.strip() for el in self._vocab_list]
    self._lowercase = lowercase
    if self._lowercase:
      self._vocab_list = [t.lower() for t in self._vocab_list]
    # Note that internally everything is 0-indexed. Padding is dealt with at the
    # end of encode and the beginning of decode.
    self._token_to_id = dict(
        zip(self._vocab_list, range(len(self._vocab_list))))
    self._oov_buckets = oov_buckets
    self._oov_token = tf.compat.as_text(oov_token)

    # Reserved tokens are all tokens that are mixed alphanum and non-alphanum.
    reserved_tokens = [t for t in self._vocab_list]

    self._decode_token_separator = decode_token_separator

  def encode(self, s):
    s = tf.compat.as_text(s)
    if self.lowercase:
      s = s.lower()
    ids = []
    for token in s.split():
      int_id = self._token_to_id.get(token, -1)
      if int_id < 0:
        int_id = self._oov_bucket(token)
        if int_id is None:
          raise ValueError("Out of vocabulary token %s" % token)
      ids.append(int_id)

    # Increment for pad id 0
    return ids#pad_incr(ids)

  def decode(self, ids):
    #ids = pad_decr(ids)

    tokens = []
    for int_id in ids:
      if int_id < len(self._vocab_list):
        tokens.append(self._vocab_list[int_id])
      else:
        tokens.append(self._oov_token)
    return self._decode_token_separator.join(tokens)

  @property
  def vocab_size(self):
    # Plus 1 for pad
    return len(self._vocab_list) + self._oov_buckets + 1

  @property
  def tokens(self):
    return list(self._vocab_list)

  @property
  def oov_token(self):
    return self._oov_token

  @property
  def lowercase(self):
    return self._lowercase

  def _oov_bucket(self, token):
    if self._oov_buckets <= 0:
      return None
    if self._oov_buckets == 1:
      return len(self._vocab_list)
    hash_val = int(hashlib.md5(tf.compat.as_bytes(token)).hexdigest(), 16)
    return len(self._vocab_list) + hash_val % self._oov_buckets

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".tokens"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    kwargs = {
        "oov_buckets": self._oov_buckets,
        "lowercase": self._lowercase,
        "oov_token": self._oov_token,
    }
    self._write_lines_to_file(filename, self._vocab_list, kwargs)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    vocab_lines, kwargs = cls._read_lines_from_file(filename)
    return cls(vocab_list=vocab_lines, **kwargs)

def _prepare_reserved_tokens(reserved_tokens):
  """Prepare reserved tokens and a regex for splitting them out of strings."""
  reserved_tokens = [tf.compat.as_text(tok) for tok in reserved_tokens or []]
  dups = _find_duplicates(reserved_tokens)
  if dups:
    raise ValueError("Duplicates found in tokens: %s" % dups)
  reserved_tokens_re = _make_reserved_tokens_re(reserved_tokens)
  return reserved_tokens, reserved_tokens_re

def _find_duplicates(els):
  seen = set()
  dups = []
  for x in els:
    if x in seen:
      dups.append(x)
    else:
      seen.add(x)
  return dups

def _re_escape(s):
  """Escape regex control characters."""
  escaped = re.sub(r"[(){}\[\].*?|^$\\+-]", r"\\\g<0>", s)
  return escaped

def _make_reserved_tokens_re(reserved_tokens):
  """Constructs compiled regex to parse out reserved tokens."""
  if not reserved_tokens:
    return None
  escaped_tokens = [_re_escape(rt) for rt in reserved_tokens]
  pattern = "(%s)" % "|".join(escaped_tokens)
  reserved_tokens_re = _re_compile(pattern)
  return reserved_tokens_re

def pad_decr(ids):
  """Strip ID 0 and decrement ids by 1."""
  if len(ids) < 1:
    return list(ids)
  if not any(ids):
    return []  # all padding.
  idx = -1
  while not ids[idx]:
    idx -= 1
  if idx == -1:
    ids = ids
  else:
    ids = ids[:idx + 1]
  return [i - 1 for i in ids]

def pad_incr(ids):
  """Add 1 to ids to account for pad."""
  return [i + 1 for i in ids]