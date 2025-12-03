from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["expert_link"]

students = db["students"]
mentors = db["mentors"]
questions = db["questions"]
