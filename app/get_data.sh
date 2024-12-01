SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"




touch"$PROJECT_ROOT/data/corpus.txt"
wget -O "$PROJECT_ROOT/data/corpus.txt" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt