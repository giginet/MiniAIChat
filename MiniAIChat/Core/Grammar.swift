protocol Grammar: Sendable, Hashable, Equatable {
    var bnf: String { get }
}

struct JSONGrammar: Grammar {
    let bnf = #"""
# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays

root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) | # escapes
    jp-char
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}

jp-char     ::= hiragana | katakana | punctuation | cjk
hiragana    ::= [ぁ-ゟ]
katakana    ::= [ァ-ヿ]
punctuation ::= [、-〾]
cjk         ::= [一-鿿]
"""#
}

struct JSONWithPrefectureGrammar: Grammar {
    let bnf = ##"""
    root ::= "[" ws01 (root-items (ws01 "," ws01 root-items)*)? ws01 "]" ws01
    root-items ::= "{" ws01 root-items-prefecture "," ws01 root-items-capital "}"
    root-items-prefecture ::= "\"prefecture\"" ":" ws01 string
    root-items-capital ::= "\"capital\"" ":" ws01 string


    value  ::= (object | array | string | number | boolean | null) ws

    object ::=
      "{" ws (
        string ":" ws value
        ("," ws string ":" ws value)*
      )? "}"

    array  ::=
      "[" ws01 (
                value
        ("," ws01 value)*
      )? "]"

    string ::=
      "\"" (string-char)* "\""

    string-char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) | jp-char # escapes

    number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
    integer ::= "-"? ([0-9] | [1-9] [0-9]*)
    boolean ::= "true" | "false"
    null ::= "null"

    # Optional space: by convention, applied in this grammar after literal chars when allowed
    ws ::= ([ \t\n] ws)?
    ws01 ::= ([ \t\n])?

    jp-char     ::= hiragana | katakana | punctuation | cjk
    hiragana    ::= [ぁ-ゟ]
    katakana    ::= [ァ-ヿ]
    punctuation ::= [、-〾]
    cjk         ::= [一-鿿]
    """##
}
