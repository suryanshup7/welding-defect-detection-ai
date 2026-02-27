## welding-defect-detection-ai
AI-powered welding defect detection system developed during my internship at Tephratec Solutions Pvt. Ltd. for a Siemens Energy use-case. Uses computer vision and machine learning, optimized for high accuracy on a small dataset, and deployed as a Streamlit web app for real-time inspection and visualization.

# Project Overview
In manufacturing, weld quality is critical for safety and performance. Traditional manual inspection is **slow, inconsistent, and prone to human error**.  

This project applies **state-of-the-art deep learning techniques** to automatically detect welding defects from images. By leveraging **YOLOv8**, the system identifies various defect types, helping industries make **data-driven quality control decisions** and improve operational efficiency.

---

## Problem Statement & Motivation
**Problem:**  
- Manual weld inspection is labor-intensive and inconsistent.  
- Subtle defects such as cracks, porosity, and incomplete fusion are often missed.  
- Poor weld quality leads to safety risks, production delays, and higher maintenance costs.  

**Motivation & Impact:**  
- Automate defect detection to **improve inspection speed and accuracy**.  
- Provide actionable insights for **industrial decision-making**.  
- Enable predictive maintenance and future process optimization.  
- Directly support companies like **Siemens Energy** in reducing faulty production and ensuring safety compliance.

---

## Architecture Overview
The solution uses **YOLOv8 (Ultralytics)** as the core model for defect detection.  

**Workflow Summary:**  
1. **Input:** High-resolution weld images from production or inspections.  
2. **Preprocessing:** Resizing, normalization, and controlled augmentation.  
3. **Detection:** YOLOv8 identifies defect locations and classifies types.  
4. **Output:** Defect labels with confidence scores; visualizations can highlight defect areas for inspection validation.  

**Training Strategy:**  
- **Phase 1:** Backbone frozen to leverage pre-trained knowledge while training the classifier head.  
- **Phase 2:** Will be executed when additional labeled data from Siemens Energy becomes available, allowing full fine-tuning for improved performance.  

This approach balances **model efficiency, accuracy, and adaptability** to industrial datasets.

---

## Results & Achievements
- **Detection Accuracy:** High performance on test images with minimal false positives.  
- **Evaluation Metrics:** Precision, Recall, and F1-score calculated per defect type.  
- **Data Visualization & Insights:**  
  - **Trend Analysis:** Historical defect data is visualized through graphs and charts to track **defect frequency over time**, helping the company identify recurring issues.  
  - **Defect Distribution:** Heatmaps and bar charts show **which types of defects occur most frequently** and in which sections of the welding process.  
  - **Operational Insights:** By combining defect locations with production metadata, the company can **pinpoint process steps or machinery contributing to defects**, enabling targeted improvements.  
  - **Decision Support:** Visualizations allow engineers and managers to **quickly grasp problem areas**, prioritize inspections, and plan **preventive maintenance** based on historical trends.  

**Business Relevance:**  
- Enables **data-driven quality control decisions**.  
- Helps reduce **faulty production and safety risks**.  
- Supports **predictive maintenance strategies** for future operations.  

> *For detailed graphs, visualizations, and analysis methodology, refer to the project notebooks and [models/README.md](models/README.md).*  

---

**Summary:** This project demonstrates a **practical AI solution for industrial quality control**, combining cutting-edge object detection techniques with a scalable, industry-ready workflow. Its design is modular, reproducible, and easily extendable for future phases, reflecting a professional AI engineer’s approach to solving real-world problems.

