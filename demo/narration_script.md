# DEFault Demo — Narration Script
**Target length:** ~4 minutes | **For:** ICST 2026 tool paper screencast

---

## [0:00 – 0:20] Scene 1 — Landing page

> "This is DEFault — a web-based tool that automatically detects and explains faults in deep neural networks written with Keras and TensorFlow.
> DEFault is based on our ICSE 2025 research paper and implements a three-stage hierarchical fault classifier.
> The interface has three panels: the code editor on the left, the analysis pipeline in the centre, and SHAP-based explanations on the right."

---

## [0:20 – 1:30] Scene 2 — Static code analysis

> "Let me paste a buggy Keras model. This is a ten-class classifier — notice the output layer uses ReLU activation instead of softmax. That's a classic fault type."

*[type code into editor]*

> "I'll name this model 'mnist_classifier' and click **Check Model** — this runs Stage 3 static analysis in about three seconds, no training needed."

*[click Check Model, wait ~3 s]*

> "DEFault flagged this as faulty with a probability of 56%, above the detection threshold.
> The SHAP waterfall on the right breaks down exactly which architectural features pushed the prediction toward 'buggy'.
> The top contributor is **Countsoftmax** — the absence of a softmax activation — which is precisely the bug we introduced.
> Below that, the insights panel gives a plain-English hint: 'Softmax usage may be incorrect for output formulation.'"

*[scroll through pipeline and taxonomy]*

> "The fault taxonomy tree in the centre highlights the flagged category — **Activation Function** under Model Faults."

---

## [1:30 – 1:45] Transition

> "Now let me show the full diagnosis pipeline — which also trains the model and collects real gradient and memory statistics."

*[click Reset]*

---

## [1:45 – 3:30] Scene 4 — Live Train & Diagnose

> "I'll paste a binary classification model. This one has another subtle bug — the output neuron uses ReLU instead of sigmoid.
> This time I'll click **Train & Diagnose**, which will actually train the model on synthetic dummy data and stream the results epoch by epoch."

*[type code, click Train & Diagnose]*

> "You can see each epoch arriving in real time — loss, validation loss, accuracy — plotted live in the training chart.
> During training DEFault also collects gradient statistics, dying-ReLU counts, and memory usage, giving it 20 dynamic features to work with — far more signal than static analysis alone."

*[epochs streaming, ~20 s]*

> "Training is complete. DEFault now runs all three stages automatically.
> Stage 1 reports a fault probability of 0.68 — above threshold, so a fault is predicted.
> Stage 2 narrows it down: the **layer** and **activation** categories are flagged.
> Stage 3 static analysis confirms it — the top SHAP feature is again the missing sigmoid, the root cause of poor convergence on a binary task."

*[scroll through results panels]*

> "The insights panel summarises the diagnosis in plain language, making results actionable for practitioners without requiring deep ML expertise."

---

## [3:30 – 3:50] Wrap-up

> "DEFault is fully open source, available at github.com/SigmaJahan/DEFault-tool, and hosted live on Hugging Face Spaces — no installation needed.
> The tool supports static-only analysis, training-history paste, and live training, and covers nine fault categories across the DEFault fault taxonomy.
> Thank you."

---

## Recording tips
- Record at **1440 × 860** so text is legible in the final video
- Use `python demo/record_demo.py` to generate the browser recording automatically
- Add this narration as a **voice-over** track in iMovie / DaVinci Resolve / ScreenFlow
- Upload to YouTube as **Unlisted** and paste the URL into the paper abstract
