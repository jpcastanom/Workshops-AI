# Workshop 4: Sistema RAG con NVIDIA NIM + Qdrant

En este cuarto workshop construimos un sistema completo de **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre la transcripción de la charla *"Intro to Large Language Models"* de Andrej Karpathy.

El objetivo no era solo hacer que un modelo respondiera preguntas, sino entender cómo combinar recuperación semántica de información con generación de lenguaje natural, y comparar el desempeño de distintos modelos de lenguaje sobre la misma base de conocimiento.

---

## Descripción

El pipeline se implementó en dos fases bien diferenciadas:

- Ingesta del documento, división en chunks, generación de embeddings e indexación en una base de datos vectorial persistente en disco.
- Carga del índice, construcción del pipeline LCEL y generación de respuestas mediante tres LLMs distintos accedidos a través de NVIDIA NIM.

La idea fue experimentar con el mismo stack RAG pero variando el modelo generativo, para poder evaluar cuál produce respuestas de mayor calidad sobre el mismo conjunto de preguntas.

---

## Archivos del Proyecto

- `RAGLLM.ipynb`: Notebook principal con las dos fases implementadas y la generación de respuestas para los tres modelos.
- `docs/intro-to-llms-karpathy.txt`: Transcripción del video de Karpathy (fuente de conocimiento del sistema).
- `docs/questions.json`: Las 50 preguntas predefinidas sobre las cuales se evalúa el sistema.
- `db/karpathy_qdrant/`: Índice Qdrant persistente generado en la Fase A.
- `llama_rag_output.json`: Respuestas generadas por Llama 4 Maverick.
- `gemma_rag_output.json`: Respuestas generadas por Gemma 3.
- `mistral_rag_output.json`: Respuestas generadas por Mistral Large 3 ✅ *(mejor modelo)*
- `README.md`: Documentación del workshop (este archivo).

---

## Objetivos de Aprendizaje

- Comprender la arquitectura RAG y por qué mejora las respuestas de un LLM.
- Implementar una base de datos vectorial persistente con Qdrant.
- Generar embeddings locales con HuggingFace sin depender de una API externa.
- Construir un pipeline de recuperación y generación usando LangChain LCEL.
- Comparar el comportamiento de distintos LLMs sobre el mismo conjunto de preguntas y contextos.

---

## Stack Tecnológico

| Componente    | Tecnología                                      |
|---------------|-------------------------------------------------|
| LLMs          | NVIDIA NIM — Llama 4, Gemma 3, Mistral Large 3  |
| Embeddings    | HuggingFace — `all-mpnet-base-v2` (local)       |
| Vector store  | Qdrant persistente en disco                     |
| Pipeline      | LangChain LCEL                                  |

---

## Fase A — Ingesta

Se cargó la transcripción completa del video de Karpathy y se dividió en chunks de 1000 caracteres con solapamiento de 200. Cada chunk fue transformado en un vector de 768 dimensiones usando el modelo `all-mpnet-base-v2` de HuggingFace ejecutado localmente. Los vectores se almacenaron en una colección Qdrant persistida en disco para no repetir este proceso costoso en cada consulta.

Esta fase se ejecuta **una sola vez**. El cliente Qdrant queda abierto y es reutilizado directamente por la Fase B sin necesidad de abrir una segunda instancia.

---

## Fase B — Pipeline RAG y Comparación de Modelos

Con el índice ya construido, se configuró un retriever con búsqueda por umbral de similitud (`score_threshold=0.3`, `k=4`) para recuperar los fragmentos más relevantes a cada pregunta. El pipeline LCEL encadena la recuperación, el formateo del contexto, el prompt y la generación de la respuesta.

Se implementaron tres pipelines idénticos en estructura, cambiando únicamente el modelo generativo:

- **Llama 4 Maverick** (`meta/llama-4-maverick-17b-128e-instruct`)
- **Gemma 3** (`google/gemma-3n-e2b-it`)
- **Mistral Large 3** (`mistralai/mistral-large-3-675b-instruct-2512`)

Cada modelo respondió las mismas 50 preguntas del archivo `questions.json`, y los resultados se guardaron en archivos JSON separados con la pregunta, la respuesta generada y los fragmentos de contexto recuperados.

---

## Resultados

Los tres pipelines fueron evaluados cualitativamente comparando la precisión, coherencia y completitud de las respuestas generadas. Tanto **ChatGPT** como **Claude** coincidieron en que **Mistral Large 3** fue el modelo que produjo las respuestas más precisas y mejor fundamentadas en el contexto recuperado. Sus respuestas mostraron mayor fidelidad a la fuente y menos tendencia a generar contenido fuera del contexto proporcionado.

El archivo entregable principal es `mistral_rag_output.json`.

---

## Uso de Herramientas de IA

Durante el desarrollo del workshop utilizamos distintas herramientas de IA como apoyo. Estas **no reemplazan el trabajo propio**, sino que sirvieron como soporte para entender conceptos, depurar errores y mejorar la documentación.

**Claude (Anthropic)**
Lo usamos principalmente como apoyo conceptual y de documentación. Nos ayudó a:
- Entender el funcionamiento interno del pipeline RAG.
- Organizar y redactar la documentación.
- Aclarar dudas sobre LangChain LCEL y Qdrant.
- Evaluar y comparar cualitativamente las respuestas de los tres modelos.

**ChatGPT (OpenAI)**
Lo utilizamos para resolver dudas técnicas puntuales:
- Debugging de errores en el entorno.
- Comandos de Git y configuración del proyecto.
- Evaluar y comparar cualitativamente las respuestas de los tres modelos.

---

## Ejecución

```bash
# Instalar dependencias
pip install openai langchain langchain-huggingface langchain-qdrant \
            qdrant-client sentence-transformers python-dotenv

# Ejecutar el notebook
jupyter notebook RAGLLM.ipynb
```

> Asegúrate de crear un archivo `.env` con la clave de NVIDIA NIM antes de ejecutar la Fase B:
> ```
> NVIDIA_API_KEY=nvapi-...
> ```

---

## Referencias

- [Intro to Large Language Models — Andrej Karpathy (YouTube)](https://www.youtube.com/watch?v=zjkBMFhNj_g) *(bajo licencia CC-BY)*
- [NVIDIA NIM — build.nvidia.com](https://build.nvidia.com)
- Documentación oficial de LangChain
- Documentación oficial de Qdrant
- Repositorio del curso (material del profesor, diapositivas y enunciados)

---

## Autores

- Juan Pablo Castaño  
- Sara Sofía Quintero  
- Karol Vanessa Cuello

*Estudiantes de Ingeniería Matemática – Universidad EAFIT*
