# Camera Calibration

Extrai os parâmetros internos da câmera para calibração.

## Configuração

- Versão do Python: 3.10.7

### Crie um ambiente de desenvolvimento virtual

- No ambiente Windows, no bash do git, execute
```bash
python -m venv pCC
```

- No ambiente Linux, execute
```bash
python3.10 -m venv pCC
```

### Ative o ambiente de desenvolvimento virtual

- No ambiente Windows, no bash do git, execute
```bash
source pCC/Scripts/activate
```

- No ambiente Linux, execute
```bash
source pCC/bin/activate
```

### Instale as dependências

```bash
pip install -r requirements.txt
```

## Como usar

Para avaliar o código, ative o ambiente de desenvolvimento e execute o comando

```bash
python calibra.py
```