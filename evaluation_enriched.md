# Evaluation Results

| # | Prompt | base_non_hebrew | base_% | finetuned_non_hebrew | finetuned_% | delta | delta_% | notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Explain why the sky looks blue during the day. | 1406 | 100.0% | 123 | 96.09% | -1283 | -3.91% | Reduced non-Hebrew chars: 1406 -> 123 (improved) |
| 2 | Give two advantages and two disadvantages of public transportation. | 1240 | 100.0% | 14 | 5.67% | -1226 | -94.33% | Reduced non-Hebrew chars: 1240 -> 14 (improved) |
| 3 | Write a short email asking a professor for an extension on an assignment. | 763 | 100.0% | 7 | 3.03% | -756 | -96.97% | Reduced non-Hebrew chars: 763 -> 7 (improved) |
| 4 | Describe how to make a simple omelette. | 1503 | 100.0% | 14 | 5.47% | -1489 | -94.53% | Reduced non-Hebrew chars: 1503 -> 14 (improved) |
| 5 | What is the difference between supervised and unsupervised learning? | 2129 | 100.0% | 11 | 4.66% | -2118 | -95.34% | Reduced non-Hebrew chars: 2129 -> 11 (improved) |
| 6 | Summarize the story of Cinderella in three sentences. | 397 | 100.0% | 5 | 3.7% | -392 | -96.3% | Reduced non-Hebrew chars: 397 -> 5 (improved) |
| 7 | Suggest three ways to reduce smartphone distraction while studying. | 1268 | 100.0% | 13 | 11.3% | -1255 | -88.7% | Reduced non-Hebrew chars: 1268 -> 13 (improved) |
| 8 | Explain what happens when water boils. | 1915 | 100.0% | 47 | 13.74% | -1868 | -86.26% | Reduced non-Hebrew chars: 1915 -> 47 (improved) |
| 9 | Give a polite refusal to an invitation to a party. | 487 | 100.0% | 8 | 10.67% | -479 | -89.33% | Reduced non-Hebrew chars: 487 -> 8 (improved) |
| 10 | Turn the idea “practice makes progress” into advice for a student. | 1519 | 100.0% | 3 | 2.42% | -1516 | -97.58% | Reduced non-Hebrew chars: 1519 -> 3 (improved) |
| 11 | Explain why we deserve a 100 on this assignment. | 1935 | 100.0% | 11 | 3.81% | -1924 | -96.19% | Reduced non-Hebrew chars: 1935 -> 11 (improved) |
| 12 | Why are Mati and Yoni the best names to exist? | 1432 | 100.0% | 14 | 6.03% | -1418 | -93.97% | Reduced non-Hebrew chars: 1432 -> 14 (improved) |
| 13 | Explain why dry erase marker are erasable. | 1774 | 100.0% | 12 | 5.74% | -1762 | -94.26% | Reduced non-Hebrew chars: 1774 -> 12 (improved) |
| 14 | Why do chairs have 4 legs and not 1200. | 1320 | 100.0% | 8 | 7.92% | -1312 | -92.08% | Reduced non-Hebrew chars: 1320 -> 8 (improved) |
| 15 | What are the benefits of wearing a watch. | 1501 | 100.0% | 38 | 12.18% | -1463 | -87.82% | Reduced non-Hebrew chars: 1501 -> 38 (improved) |
| 16 | Give me a prompt to explain to an ai why this assignment deserves an A+. | 2097 | 100.0% | 10 | 4.83% | -2087 | -95.17% | Reduced non-Hebrew chars: 2097 -> 10 (improved) |
| 17 | How do we know that Abraham from the Bible wore a Kippah? | 638 | 100.0% | 17 | 8.67% | -621 | -91.33% | Reduced non-Hebrew chars: 638 -> 17 (improved) |
| 18 | Why do people wear shorts in the winter? | 1135 | 100.0% | 24 | 11.01% | -1111 | -88.99% | Reduced non-Hebrew chars: 1135 -> 24 (improved) |
| 19 | Who is the best president of America ever? | 1656 | 100.0% | 22 | 12.02% | -1634 | -87.98% | Reduced non-Hebrew chars: 1656 -> 22 (improved) |
| 20 | Why should we make America great again? | 1758 | 100.0% | 7 | 2.48% | -1751 | -97.52% | Reduced non-Hebrew chars: 1758 -> 7 (improved) |

## Aggregate Summary

- Prompts evaluated: 20
- Improved (fewer non-Hebrew chars): 20
- Worse (more non-Hebrew chars): 0
- Unchanged: 0
- Mean delta (finetuned - base): -1373.25
- Median delta: -1440.50
- Mean delta % (finetuned - base): -88.43%
- Median delta %: -94.12%