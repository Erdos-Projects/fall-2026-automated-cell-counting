# Weekly Check-ins

Each week's check-in lives in its own file, `checkin_weekN.md`. This page is the index, and below it are the requirements for each week as given by the course.

| Week | Topic | Status | Check-in |
|---|---|---|---|
| 1 | Project scope, data sources, stakeholders, KPIs | ✅ Submitted | [checkin_week1.md](checkin_week1.md) |
| 2 | Exploratory data analysis; preprocessing | ✅ Ready | [checkin_week2.md](checkin_week2.md) |
| 3 | Model architecture: baseline and deep learning | Upcoming | — |
| 4 | Baseline model performance | Upcoming | — |
| 5 | Preprocessed data; deep learning model, iteration 1 | Upcoming | — |
| 6 | Deep learning model, iteration 2 | Upcoming | — |
| 7 | Final project | Upcoming | — |

Related notes: [`../plan.md`](../plan.md) (roadmap), [`../eda_summary.md`](../eda_summary.md) (EDA with figures), [`../project.md`](../project.md) (scope and KPIs).

---

## Requirements by week

### Week 1
- Describe your project scope (1 paragraph)
- Describe your data sources and include links if they exist
- List project stakeholders: the people who are most invested in your project and will make decisions based on your results
- List Key Performance Indicators (KPIs): metrics that will indicate project success. Examples: 20% sensitivity above baseline model, 50% less compute time

### Week 2
2. Basic exploratory data analysis of data; discussion of preprocessing techniques needed
- Have your dataset uploaded to your workspace
- Do some basic exploratory data analysis (EDA) on the data: type of EDA will depend on data type, such as continuous, ordinal, image, or audio data
- List insights into data preprocessing techniques needed. Examples:
  1. Should all data be included in the model or is inclusion / exclusion criteria needed?
  2. What types of transformations / preprocessing needs to be done before they can be put in the model?
  3. How much data will go in the train / valid / test sets? Is this enough that deep learning won't overfit (it overfits way more easily than traditional ML)

### Week 3
3. Describe model architecture decisions for baseline model and deep learning model
- Based on your data exploration from last week, examine various model architectures
- Start with a baseline model. This can be a non-deep learning method or an existing published model you'll compare your model to
- Compare different deep learning techniques. Discussions can consider expected performance level, compute needed, inference time, and past research. Example: if you're doing NLP, what is most promising for your purposes and why: RNN, LSTM, transformers.

### Week 4
4. Baseline model performance
- Use your data in the proposed baseline model from last week.
- List the performance metrics and any useful diagnostic figures
- Describe where the model worked well and where it didn't work well. Example: Model inference time took <5 minutes. Audio classification accuracy was 90% for "clean" data (no background noise), but was lower in accuracy for audio data with background noise. Across different background noise types (e.g., car horns, static, other people talking), mean accuracy was 60%, with a standard deviation of 25%

### Week 5
5. Fully preprocessed data (for DL model) Deep Learning model performance (iteration 1); discussion of what went wrong and how to fix it
- Ensure your data is preprocessed correctly for the deep learning model you're using. Example: for image data, rotation, progressive resizing, mixup
- Put your preprocessed data through the deep learning architecture.
- List the performance metrics and any useful diagnostic figures
- Describe where the model worked well and where it didn't work well

### Week 6
6. Deep Learning model performance (iteration 2); discussion of what went wrong and how to fix it
- Discuss how your strategy changed from last week based on the results. Examples: Did you move to a different model architecture because you were overfitting? Did you add data transformation steps to your preprocessing pipeline?
- Put your preprocessed data through the revised deep learning model architecture.
- List the performance metrics and any useful diagnostic figures
- Describe where the model worked well and where it didn't work well

### Week 7
7. Final project due
- Annotated GitHub repository, so people can follow your logic
  - Nice Markdown file, docstrings, comments in the code. No "dead code"
  - Someone not on your team should be able to run the entire codebase, since all data is public
  - Strong preference for preprocessing / modeling pipelines with .py files and results script in an .ipynb format
- Executive summary of your project results and implications
  - 1 page maximum
  - Focus on business impact, instead of only on model accuracy. The goal is to show how your model will help the stakeholders make business decisions. If your model works well, but will be impractical because, say, the lag time is too long to use in a production setting, say so.
- 5-min pre-recorded PowerPoint presentation detailing project process from start to finish
  - Overall project vision, methodology, and results
  - Focus on visuals and impact, instead of too many technical details
  - Take them on a journey, showing how you problem-solved
