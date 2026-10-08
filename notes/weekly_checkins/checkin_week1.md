# Week 1 Check-in: Project Scope, Data, Stakeholders, KPIs

[← Check-in index](required_checkin_items.md) · Status: ✅ submitted

## Describe your project scope (1 paragraph)

With innovations in the automation of many processes in the medical research field, the task of counting cells in microscopy images remains tediously manual and time-consuming. We aim to develop a deep learning model, Cell Image Counting, that will take in microscopy images and output the estimated cell count. The model will be trained with images from a variety of cell sources, as we intend for a wide application to different kinds of cell samples, and will involve a thorough evaluation of performance on images with high cell density, overlapping cells, and imaging quality.

## Describe your data sources and include links if they exist

We have a range of datasets consisting of images from different cell samples and imaging types, we hope this accounts for the variation that will be present in the real-life application of this model, but we also continue to search for other relevant datasets.

- **Synthetic Cell Images and Maks:** A dataset consisting of 20,400 black and white cell images with different degrees of blur applied, to account for the variety of image quality. Each image is tagged with the number of cells and the degree of applied blur, with the non-blurred images serving as the ground truth.
- **Cell Counting Computer Vision Model:** A dataset of 1,200 images with different color backgrounds and cell count number.
- **Cell-Counting Computer Vision Model 2:** A dataset with 531 images of different imaging varieties and types of cells with cell count number and a cell-wise classification of normal or abnormal cells.
- **Blood Cell Count and Detection (BCCD):** A dataset with 874 images with detailed annotations like the type of blood cell and other diagnostic features, and can be used for counting different types of blood cells.
- **CoNIC:** Model that performs segmentation, classification, and counting of six different types of nuclei.

## List project stakeholders

Medical companies or pathology laboratories would be interested in the success of this project, along with research labs (private and universities). The ability to count cells has many applications, from cell culture monitoring to disease research, but specific interest will be dependent on the model performance on these different varieties.

## List Key Performance Indicators (KPIs)

We will begin our project with available open-source models and by using the model performance as baseline, will investigate the scope for improvement to the baseline. Our main performance metric will be accuracy and R2,  but we will also consider comparison between computational efficiency and resources allocation for cell counting with the model and compare to the manual process.

---

> **Update (week 2, 2026-10-07):** the 1,200-image "Cell Counting Computer Vision Model" set is 500 unique images plus augmented training copies. We use the 500 originals. See [`checkin_week2.md`](checkin_week2.md).
