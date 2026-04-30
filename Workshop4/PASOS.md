# 🛠️ Pasos para Resolver el Workshop 4 - Sistema RAG

## Paso 1: Configurar el entorno

1. Instalar dependencias necesarias:
   ```bash
   pip install langchain langchain-google-genai langchain-huggingface langchain-community chromadb sentence-transformers python-dotenv
   ```

2. Crear el archivo `.env` en la raíz del Workshop4:
   ```bash
   GOOGLE_API_KEY=tu_clave_aqui
   ```

---

## Paso 2: Crear el script de ingesta (`ingest.py`)

Crea el archivo `ingest.py` que carga el documento, lo divide en chunks y crea el vector store persistente en `db/karpathy_chroma`.

- Usa `TextLoader` para cargar `docs/intro-to-llms-karpathy.txt`
- Usa `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200`
- Usa `HuggingFaceEmbeddings` con el modelo `sentence-transformers/all-mpnet-base-v2`
- Usa `Chroma.from_documents()` con `persist_directory="db/karpathy_chroma"`
- Llama a `vector_store.persist()` al final

Ejecutar **una sola vez**:
```bash
python ingest.py
```

---

## Paso 3: Crear el pipeline RAG (`rag_pipeline.py`)

Crea el archivo `rag_pipeline.py` que:

- Carga las variables de entorno con `load_dotenv()`
- Inicializa el LLM: `ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")`
- Carga los mismos embeddings (`all-mpnet-base-v2`)
- Carga el vector store desde disco con `Chroma(persist_directory=..., embedding_function=embeddings)`
- Construye el pipeline con `RetrievalQA.from_chain_type(..., return_source_documents=True)`

> ⚠️ Corregir el typo del README: `as_retris_ever()` → `as_retriever()`

---

## Paso 4: Generar respuestas a las 50 preguntas (`generate_answers.py`)

Crea el archivo `generate_answers.py` que:

1. Carga las preguntas desde `docs/questions.json`
2. Itera sobre cada pregunta usando el pipeline RAG
3. Guarda los resultados en `my_rag_output.json` con el formato:
   ```json
   [
     {
       "question": "...",
       "answer": "...",
       "contexts": ["fragmento 1", "fragmento 2"]
     }
   ]
   ```

Ejecutar:
```bash
python generate_answers.py
```

---

## Paso 5: Verificar y entregar

- [ ] Verificar que `my_rag_output.json` contiene las 50 preguntas con sus respuestas y contextos
- [ ] No modificar las preguntas originales del archivo `questions.json`
- [ ] Entregar:
  - `ingest.py`
  - `rag_pipeline.py` (o el script equivalente)
  - `generate_answers.py`
  - `my_rag_output.json`

---

## Estructura final esperada del proyecto

```
Workshop4/
├── .env
├── docs/
│   ├── intro-to-llms-karpathy.txt
│   └── questions.json
├── db/
│   └── karpathy_chroma/        ← generado por ingest.py
├── ingest.py
├── rag_pipeline.py
├── generate_answers.py
└── my_rag_output.json           ← generado por generate_answers.py
```
