# 🖊️ Text2Sketch: Generate 2D Floor Plans from Text

**Text2Sketch** is an AI-powered system that takes in natural language descriptions like “a 3-room apartment with a kitchen and two bathrooms” and generates 2D floor plans using a GAN-based architecture. This project merges the domains of NLP and Computer Vision to automate architectural sketching.

---

##  Project Overview

- **Goal:** Convert descriptive text inputs into structured 2D floor plan images.
- **Approach:** Utilize a text encoder combined with a GAN to generate sketches.
- **Team Members:** Rushil Shah, Vivek, Kunal, Vanshika

---

## ⚙️ Tools & Technologies Used

| Category             | Tools/Technologies               |
|----------------------|----------------------------------|
| Programming Language | Python                           |
| Deep Learning        | PyTorch, GAN (DCGAN)             |
| Data Handling        | NumPy, JSON                      |
| Visualization        | Matplotlib                       |
| Version Control      | Git, GitHub                      |
| OS & IDE             | Windows 10, Linux, VS Code       |
| Optional Extensions  | Jupyter Notebooks                |

---

## 🧩 Project Structure

```
Text2Sketch/
├── data_preprocessing.py     # Preprocessing & augmentation (Vivek)
├── model_architecture.py     # GAN with text encoder (Rushil)
├── train_model.py            # Training loop & custom loss (Kunal)
├── visualize_output.py       # Visualizing floor plans (Vanshika)
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── outputs/                  # Generated floor plans
```

---

## 🛠️ How to Run the Project

### Step-by-Step Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/text2sketch.git
   cd text2sketch
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset:**
   - Place floor plan images and corresponding JSON descriptions into the `data/` directory.

4. **Run Preprocessing:**
   ```bash
   python data_preprocessing.py
   ```

5. **Train the Model:**
   ```bash
   python train_model.py
   ```

6. **Generate & Visualize Output:**
   ```bash
   python visualize_output.py
   ```

---

## Submission Timeline & Project Planning

Although no formal project tracking tools were used, the team coordinated tasks manually and used Git commit timestamps to track progress.

| **Week** | **Tasks Completed**                                                   |
|----------|-----------------------------------------------------------------------|
| Week 1   | Finalized the project idea and gathered initial resources             |
| Week 2   | Prepared the dataset and extracted JSON labels                        |
| Week 3   | Implemented data preprocessing and augmentation                       |
| Week 4   | Built the GAN model and integrated the text encoder                   |
| Week 5   | Trained the model and performed hyperparameter tuning                 |
| Week 6   | Visualized outputs, tested results, and completed documentation       |

---

## Innovation & Individual Contributions

| Name     | Contribution                                            |
|----------|---------------------------------------------------------|
| Rushil   | Text encoder integration & GAN model design             |
| Vivek    | JSON label extraction and data augmentation             |
| Kunal    | Training loop implementation and custom loss functions  |
| Vanshika | Output visualization module and evaluation metrics      |

---

## 🧑‍💻 Git Commands Used

- `git init`                  : Initialize a local Git repository
- `git clone <repo_url>`      : Clone the remote repository
- `git status`                : Show the working directory status
- `git add <file>` / `git add .` : Stage changes for commit
- `git commit -m "msg"`     : Commit staged changes with message
- `git pull origin main`      : Fetch and merge changes from remote
- `git push origin main`      : Push local commits to remote
- `git log`                   : Show commit history
- `git branch`                : List branches
- `git remote -v`             : Show remote URLs

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Thanks to our faculty guide and resources like Hugging Face and PyTorch tutorials for their invaluable support.
