# Thesis Project

This repository contains the full implementation of a robotic perception and motion pipeline developed as part of a thesis project. The system processes camera input from a robot, detects and interprets whiteboard content, builds a graph representation, plans a path, and executes motion commands.

---

## 📁 Project Structure

### `/dataset`
Contains the collected image data used for experiments.  
- Includes screenshots captured from the robot’s camera during operation  
- Used as input for testing and evaluation of the pipeline  

---

### `/debug`
Contains intermediate outputs used during development. 
- Used to analyze system behavior at different stages  
- Helpful for troubleshooting detection, graph building, and planning issues  

---

### `/robot_config`
Configuration files and parameters for the robot system.  
- Based on the NicoIK project from the FMPH, Comenius University  

---

## 🔧 Main Pipeline Stages

### 1. Whiteboard Detection
- Detects and localizes the whiteboard in the camera frame  

### 2. Graph Construction
- Converts detected visual information into a structured graph representation  

### 3. Path Planning
- Computes a path based on the constructed graph  

### 4. Motion Execution
- Translates planned paths into robot commands  

---

## 📊 `/results/close`

- Contains final experiment outputs as well as scripts for analysis

---

## 🧪 Purpose

This project demonstrates an end-to-end robotic pipeline combining:
- Computer vision (whiteboard detection)
- Graph-based reasoning
- Path planning algorithms
- Real-world robot motion execution  

---

## 🏛️ Academic Context

Developed as part of a thesis project in collaboration with the FMPH, Comenius University, using components from the NicoIK robotics framework.
