# FloodVision 🌊
### Flood Risk Prediction and Disaster Awareness Platform Using Machine Learning

> A data-driven decision support platform that analyzes environmental and geographic conditions to estimate flood risk levels, assess potential impact, and promote disaster preparedness.

> **Note:** FloodVision is a predictive and educational platform. It is not a real-time alerting system.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Prediction Parameters](#prediction-parameters)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Social Impact](#social-impact)
- [Future Scope](#future-scope)

---

## Overview

FloodVision is a Machine Learning–based Flood Risk Prediction and Impact Assessment platform. Users provide environmental and geographic inputs, and the trained ML model predicts flood risk levels along with potential impact severity. The platform also serves as an awareness tool — offering safety guidelines, emergency contacts, NGO resources, and interactive flood-prone area maps.

---

## Objectives

1. Predict flood risk using Machine Learning
2. Estimate possible impact severity and damage (in INR)
3. Increase disaster awareness among users and communities
4. Provide actionable safety guidelines (before, during, and after floods)
5. Share emergency contacts and NGO resources
6. Visualize flood-prone areas on interactive maps

---

## Key Features

### 🔴 Flood Risk Prediction
Users input environmental values and the ML model returns a predicted flood risk level based on multiple geographic and rainfall features.

### 📊 Impact Assessment
Estimates the severity of a potential flood event and outlines possibly affected zones, including estimated damage in INR.

### 🗺️ Interactive Impact Maps
Map-based visualization of flood-prone regions powered by Leaflet, helping users understand geographic risk distribution at a glance.

### 📚 Flood Awareness Module
Informational content covering flood facts, common causes, early warning indicators, and general preparedness knowledge.

### 🚨 Emergency Support Information
Curated list of important emergency contact numbers and links to NGOs and relief organizations for rapid access during crisis situations.

### 🛡️ Safety Guidelines
Step-by-step guidance organized into three phases: **Before Flood**, **During Flood**, and **After Flood**.

---

## Prediction Parameters

The ML model uses the following environmental and geographic features:

| Feature | Purpose |
|---|---|
| Rainfall — 1-day average | Detects immediate flood triggers |
| Rainfall — 7-day average | Captures short-term water accumulation |
| Rainfall — 1-month average | Reflects long-term ground saturation |
| Latitude & Longitude | Precise location-based risk assessment |
| Nearby Water Bodies | Identifies overflow risk from rivers and lakes |
| Elevation | Low-lying areas carry higher flood probability |
| Geographic Data | Captures land characteristics of the region |
| Impervious Factor | Measures proportion of non-absorbent surfaces |
| Land Cover Index | Identifies land type; determines water absorption and runoff behavior |

---

## System Architecture

```
┌─────────────────────────────┐
│       Data Input Layer       │
│  User-provided rainfall and  │
│      geographic inputs       │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│     ML Processing Layer      │
│  Feature engineering +       │
│  trained prediction model    │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│    Risk Assessment Engine    │
│  Flood risk probability +    │
│      severity scoring        │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│     Visualization Layer      │
│  Interactive map-based       │
│    flood impact display      │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   Awareness & Support Module │
│  Safety guidelines, contacts │
│      and NGO information     │
└─────────────────────────────┘
```

---

## Technologies Used

| Category | Technology |
|---|---|
| Language | Python |
| ML Framework | Scikit-learn |
| ML Models | Classification & Regression |
| Training Data | Geospatial datasets |
| Mapping | Leaflet |
| Web Framework | Streamlit |

---

## Social Impact

Floods are among the most destructive and frequently occurring natural disasters. FloodVision contributes to community resilience by:

- Improving individual and community-level flood preparedness
- Supporting awareness and public education around flood risk
- Helping users identify and understand high-risk zones
- Encouraging data-driven thinking in disaster planning

---

## Future Scope

- 🌐 Real-time weather API integration
- 🔔 Automated flood alert notifications
- 🛰️ Satellite imagery incorporation
- 📱 Mobile application version
- 🏛️ Government disaster management system integration

---

## Conclusion

FloodVision demonstrates how Machine Learning and geographic intelligence can meaningfully improve flood risk understanding and community awareness. By making predictive insights accessible to everyday users, it serves as a practical support tool for better disaster preparedness and informed decision-making.

---

> ⚠️ **Disclaimer:** FloodVision is intended for educational and awareness purposes only. It is not a substitute for official government flood warnings or emergency services.
