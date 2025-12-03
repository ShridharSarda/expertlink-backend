# backend/ml/generate_synthetic_mentors.py
import os
import random
import json
import re
from db import SessionLocal
from models import User
from utils import hash_password

BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, "..", "data", "synthetic_questions.csv")
VOCAB = os.path.join(BASE, "subject_vocab.json")

random.seed(42)

# number of mentors to generate
NUM_MENTORS = 120   # adjust if you want more/less

# list of 120 human-like names (as requested)
NAMES = [
 "Arjun Mehta","Kavya Desai","Rohan Kulkarni","Tanvi Iyer","Milan Joshi","Priya Shah","Dev Mishra","Sneha Kamat","Nikhil Saxena","Aditi Bansal",
 "Vivek Rathi","Rhea Malhotra","Karthik Rao","Neha Sinha","Pranav Bhat","Shruti Nair","Manish Kapoor","Ananya Ghosh","Pratik Sharma","Ishita Jain",
 "Arvind Sagar","Reema Verma","Jay Patel","Vidhi Trivedi","Vinay Kulshreshtha","Pooja Shinde","Harsh Vora","Meera Kaushik","Anant Jadhav","Ritu Chopra",
 "Kabir Banerjee","Sanjana Reddy","Aditya Chauhan","Pallavi Kaur","Suresh Menon","Jhanvi Gupta","Omkar Pawar","Simran Khurana","Abhishek Rana","Gayatri More",
 "Yash Mallick","Suhani Agarwal","Rohit Saluja","Namrata Pillai","Sameer Dutta","Deeksha Acharya","Deepak Varma","Shalini Sood","Hriday Khandelwal","Muskan Vatsa",
 "Ritesh Pal","Mitali Sengupta","Darshan Solanki","Heena Rawat","Tanishq Lodha","Kajal Mhatre","Parth Kohli","Radhika Rane","Kiran Bhoir","Shreya Mohanty",
 "Siddharth Rao","Kritika Bedi","Arnav Shetty","Mahima Yadav","Chirag Dholakia","Tanushree Rao","Hrithik Purohit","Sakshi More","Tejas Nikam","Ayesha Khan",
 "Vikas Chatterjee","Riddhi Bhatt","Aaryan Kulkarni","Myra Jain","Ishaan Pandey","Divya Chouhan","Keshav Bharadwaj","Natasha Fernandes","Aarav Tandon","Sana Hussain",
 "Advaith Rao","Isha Patil","Ritam Bhattacharya","Mehul Dalal","Tara Venkatesan","Rohan Mendonca","Saisha Thomas","Varun Deshmukh","Zoya Merchant","Abhinav Saxena",
 "Dhruv Suri","Palak Thakkar","Shaurya Oberoi","Kiara Menon","Atharv Sinha","Nysa Sharma","Lakshya Malhotra","Samaira Joshi","Reyansh Nair","Amaya Khanna",
 "Arjit Goel","Vaani Chaudhary","Aarush Kapoor","Riya Narang","Yuvraj Anand","Siya Bhave","Harshit Lamba","Dia Rane","Arnav Saxena","Maahi Sheth",
 "Vedant Kulkarni","Anvi Deshmukh","Aarav Acharya","Shanaya Gokhale","Hridaan Rao","Keya Gupta","Advait Joshi","Maira Paul","Atharva Shinde","Mira Pal"
]

# helper: read subject vocab
if os.path.exists(VOCAB):
    subject_vocab = json.load(open(VOCAB, "r", encoding="utf-8"))
else:
    subject_vocab = {}

available_subjects = list(subject_vocab.keys()) or ["math", "physics", "chemistry", "cs"]

# sample subject combination generator
def sample_subjects():
    k = random.choices([1,2,3], weights=[70,25,5])[0]
    return ",".join(sorted(random.sample(available_subjects, min(k, len(available_subjects)))))

# sample solved_keywords maker: pick diverse tokens per subject
def make_solved_keywords(subjects_csv, per_sub=30):
    toks = []
    for s in [x.strip().lower() for x in subjects_csv.split(",")]:
        pool = subject_vocab.get(s, [])
        if not pool:
            continue
        take = min(len(pool), per_sub)
        # pick random slice for diversity
        start = random.randint(0, max(0, len(pool)-take))
        sample = pool[start:start+take]
        # also add some sampled tokens to increase diversity
        extra = random.sample(pool, min(len(pool), max(0, per_sub//4)))
        toks += sample + extra
    # shuffle and dedupe while preserving first-seen order
    seen = set()
    out = []
    for t in toks:
        if not t:
            continue
        tt = t.strip().lower()
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    # ensure we don't return enormous lists; keep reasonable length
    return ",".join(out[:per_sub * len(subjects_csv.split(","))])

# sanitize name -> email
def make_email(name, existing_emails):
    # create firstname.lastname style
    parts = re.split(r"\s+", name.strip().lower())
    if len(parts) == 1:
        base = parts[0]
    else:
        base = f"{parts[0]}.{parts[-1]}"
    # strip non-alphanum/dot
    base = re.sub(r"[^a-z0-9\.]", "", base)
    # ensure unique by appending number if needed
    email = f"{base}@demo.local"
    idx = 1
    while email in existing_emails:
        email = f"{base}{idx}@demo.local"
        idx += 1
    existing_emails.add(email)
    return email

def generate_demo_mentors(n=NUM_MENTORS, prefix_names=None):
    db = SessionLocal()
    created = 0
    names = prefix_names[:] if prefix_names else NAMES[:]
    # if fewer names than n, repeat names with suffix
    if len(names) < n:
        # expand by adding numeric suffixes
        orig = names[:]
        i = 1
        while len(names) < n:
            for nm in orig:
                names.append(f"{nm} {i}")
                if len(names) >= n:
                    break
            i += 1
    random.shuffle(names)
    existing_emails = set()
    batch = []
    try:
        for i in range(n):
            name = names[i].strip()
            subj = sample_subjects()
            solved = make_solved_keywords(subj, per_sub=30)
            # create a clean unique email
            email = make_email(name, existing_emails)
            exp = random.randint(0, 10)
            # boolean/active flags if model supports them
            user = User(
                name=name,
                email=email,
                password=hash_password("pass"),
                role="mentor",
                subjects=subj,
                experience_years=exp,
                solved_keywords=solved,
                balance=0.0,
                is_active=True
            )
            db.add(user)
            created += 1
            # commit in batches for stability
            if created % 50 == 0:
                db.commit()
        db.commit()
    except Exception as e:
        db.rollback()
        print("Error creating mentors:", e)
    finally:
        db.close()
    print("Created mentors:", created)

if __name__ == "__main__":
    generate_demo_mentors()
