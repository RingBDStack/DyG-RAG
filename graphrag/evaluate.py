"""
Evaluation module: Calculate evaluation metrics for GraphRAG query results

This evaluation code is refer to the implementation from:
@https://github.com/JayLZhou/GraphRAG
"""
import os
import json
import re
import string
import logging
import pandas as pd
import csv
import io
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

class Evaluator:
    """Evaluation class that implements various metric calculations"""
    
    def parse_multiple_answers(self, answer_str):
        """
        Parse multiple answers, handling quoted strings that may contain commas
        
        Args:
            answer_str: Answer string that may contain multiple answers
            
        Returns:
            list: List of individual answers
        """
        answer_str = answer_str.strip()
        
        # Handle | separator first (simpler case)
        if "|" in answer_str:
            return [ans.strip() for ans in answer_str.split("|")]
        
        # Check if string contains quotes, indicating CSV-like format
        if '"' in answer_str:
            try:
                # Use CSV reader to properly parse quoted strings
                csv_reader = csv.reader(io.StringIO(answer_str))
                answers = next(csv_reader)
                return [ans.strip() for ans in answers if ans.strip()]
            except Exception:
                # If CSV parsing fails, fall back to simple comma split
                pass
        
        # Check for comma separation (simple case without quotes)
        if "," in answer_str:
            return [ans.strip() for ans in answer_str.split(",")]
        
        # Single answer
        return [answer_str]
    
    def get_label_pred_list(self, df, pred_col, label_col):
        """Get prediction and label lists"""
        pred_list = df[pred_col].tolist()
        label_list = df[label_col].tolist()
        return label_list, pred_list
    
    def normalize_answer(self, s):
        """Normalize answer string, remove punctuation, spaces and redundant information"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the|An|The)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def eval_accuracy(self, prediction, ground_truth):
        """Calculate accuracy using inclusion relationship rather than exact match"""
        # Normalize Yes/No to lowercase
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        normalized_pred = self.normalize_answer(prediction)
        normalized_gt = self.normalize_answer(ground_truth)
        
        # Use inclusion relationship, if standard answer appears in prediction, consider it correct
        if normalized_gt in normalized_pred:
            return 1
        else:
            return 0
    
    def f1_score(self, prediction, ground_truth):
        """Calculate F1 score"""
        # Normalize Yes/No to lowercase
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        # If prediction or ground truth is empty, return 0
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0, 0, 0
            
        # Calculate overlapping tokens
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        # If no overlap, F1 is 0
        if num_same == 0:
            return 0, 0, 0
            
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1, precision, recall
    
    def exact_match_score(self, prediction, ground_truth):
        """Calculate exact match score"""
        # Normalize Yes/No to lowercase
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def eval_results(self, results_file):
        """Evaluate results file, calculate various metrics"""
        # Load results file
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract results list
        results = data.get('results', [])
        
        # Filter successful queries
        successful_results = [r for r in results if r.get('status') == 'success']
        
        # 计算 query_time 平均值（只统计成功的）
        avg_query_time = 0
        if successful_results:
            total_query_time = sum(r.get('query_time', 0) for r in successful_results)
            avg_query_time = total_query_time / len(successful_results)
        
        # Create DataFrame
        df = pd.DataFrame(successful_results)
        
        # Ensure necessary columns exist
        if 'answer' not in df.columns or 'golden_answer' not in df.columns:
            logger.error("Results file missing required columns: 'answer' or 'golden_answer'")
            return None
            
        # Rename columns to fit evaluation functions
        df = df.rename(columns={'answer': 'output', 'golden_answer': 'answer'})
        
        # Perform evaluation
        res_dict, evaluated_df = self.short_eval(df)
        
        # 加入 avg_query_time 到评估结果
        res_dict['avg_query_time'] = avg_query_time
        
        # Save evaluation results
        # output_eval_file = results_file.replace('.json', '_eval.json')
        output_eval_file = './ComplexTR/results_mode-naive_topk-20_eval.json'
        
        # Merge metadata and evaluation results
        eval_data = {
            "metadata": data.get('metadata', {}),
            "eval_metrics": res_dict,
            "per_question_metrics": evaluated_df[['question_id', 'question', 'output', 'answer', 'accuracy', 'f1', 'precision', 'recall', 'em']].to_dict('records')
        }
        
        # Save evaluation results
        with open(output_eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Evaluation completed! Results saved to: {output_eval_file}")
        
        return res_dict, evaluated_df
        
    def short_eval(self, df: pd.DataFrame):
        """Calculate short answer evaluation metrics"""
        # Initialize result lists
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []

        # Get prediction and label lists
        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")

        # Calculate metrics for each question
        for prediction, answer in zip(pred_list, label_list):
            # Ensure prediction and answer are strings
            prediction = str(prediction) if prediction is not None else ""
            answer = str(answer) if answer is not None else ""
            
            # Handle possible separators for predictions
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)

            # Handle multiple answer formats using intelligent parsing
            answer_list = self.parse_multiple_answers(answer)
            
            # For multiple answers, require ALL answers to be covered
            if len(answer_list) == 1:
                # Single answer case - use original logic
                single_answer = answer_list[0].strip()
                accuracy = self.eval_accuracy(prediction_str, single_answer)
                f1, prec, recall = self.f1_score(prediction_str, single_answer)
                em = self.exact_match_score(prediction_str, single_answer)
            else:
                # Multiple answers case - require all to be covered
                all_correct = True
                covered_answers = 0
                total_answers = 0
                
                # Check if prediction contains ALL answers (for accuracy)
                # and count covered answers (for recall)
                for single_answer in answer_list:
                    single_answer = single_answer.strip()
                    if not single_answer:  # Skip empty answers
                        continue
                    
                    total_answers += 1
                    
                    # Check accuracy for each answer
                    single_accuracy = self.eval_accuracy(prediction_str, single_answer)
                    if single_accuracy == 0:
                        all_correct = False
                    else:
                        covered_answers += 1
                
                # Calculate final metrics
                accuracy = 1 if all_correct else 0
                
                # For multiple answers with distributed coverage:
                # - Precision: covered_answers / total_answers (same as recall in this case)
                # - Recall: covered_answers / total_answers 
                # - F1: harmonic mean of precision and recall
                if total_answers == 0:
                    f1, prec, recall = 0, 0, 0
                else:
                    recall = covered_answers / total_answers
                    # For this use case, precision equals recall since we're measuring coverage
                    prec = recall
                    f1 = recall  # Since prec == recall, F1 also equals recall
                
                # For exact match, use more strict comparison
                # Check if each answer has an exact match (not just inclusion)
                exact_matches = 0
                for single_answer in answer_list:
                    single_answer = single_answer.strip()
                    if not single_answer:
                        continue
                    
                    single_em = self.exact_match_score(prediction_str, single_answer)
                    if single_em == 1:
                        exact_matches += 1
                
                # EM = 1 only if ALL answers have exact matches
                em = 1 if exact_matches == total_answers and total_answers > 0 else 0
            
            # Add scores to result lists
            em_list.append(em)
            f1_list.append(f1)
            precision_list.append(prec)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

        # Calculate overall metrics
        accuracy = sum(accuracy_list) * 100 / len(accuracy_list) if accuracy_list else 0
        f1 = sum(f1_list) * 100 / len(f1_list) if f1_list else 0
        pre = sum(precision_list) * 100 / len(precision_list) if precision_list else 0
        recall = sum(recall_list) * 100 / len(recall_list) if recall_list else 0
        em = sum(em_list) * 100 / len(em_list) if em_list else 0

        # Add metrics to DataFrame
        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precision_list
        df["recall"] = recall_list
        df["em"] = em_list

        # Create result dictionary
        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "em": em,
        }

        # Print results
        logger.info("Evaluation results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {pre:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 score: {f1:.4f}")
        logger.info(f"Exact Match (EM): {em:.4f}")

        return res_dict, df

def run_evaluation(results_file):
    """Run evaluation process
    
    Args:
        results_file: Results file path
        
    Returns:
        dict: Evaluation metrics
    """
    # Create evaluator
    evaluator = Evaluator()
    
    # Check if file exists
    if not Path(results_file).exists():
        logger.error(f"Results file {results_file} does not exist!")
        return None
    
    logger.info(f"Starting evaluation of results file: {results_file}")
    
    # Perform evaluation
    metrics, _ = evaluator.eval_results(results_file)
    
    # Print summary
    logger.info(f"Evaluation completed! Overall F1: {metrics['f1']:.2f}, Exact Match: {metrics['em']:.2f}")
    
    # Return evaluation metrics
    return metrics

# Direct call example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG query results")
    parser.add_argument("--results-file", type=str, required=True, help="Query results file path")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluation_results = run_evaluation(args.results_file)
    
    if evaluation_results:
        print("Evaluation metrics:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}") 