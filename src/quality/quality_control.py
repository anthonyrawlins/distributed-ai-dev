#!/usr/bin/env python3
"""
Quality Control and Code Review System
Multi-agent validation and review workflows for distributed development
"""

import asyncio
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ReviewType(Enum):
    CODE_REVIEW = "code_review"
    PERFORMANCE_VALIDATION = "performance_validation"
    SECURITY_AUDIT = "security_audit"
    INTEGRATION_TEST = "integration_test"

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_CHANGES = "needs_changes"
    REJECTED = "rejected"

@dataclass
class ReviewResult:
    reviewer_id: str
    review_type: ReviewType
    status: ReviewStatus
    score: float  # 0-10
    feedback: str
    suggestions: List[str]
    blocking_issues: List[str]

@dataclass
class CodeSubmission:
    id: str
    task_id: str
    agent_id: str
    code: str
    language: str
    description: str
    files_modified: List[str]
    tests_included: bool = False
    performance_data: Optional[Dict] = None

class QualityController:
    def __init__(self):
        self.submissions: Dict[str, CodeSubmission] = {}
        self.reviews: Dict[str, List[ReviewResult]] = {}
        self.approval_threshold = 7.0  # Minimum score for approval
        self.required_reviewers = 2  # Minimum number of reviewers
    
    async def submit_for_review(self, submission: CodeSubmission) -> str:
        """Submit code for multi-agent review"""
        self.submissions[submission.id] = submission
        self.reviews[submission.id] = []
        
        print(f"Submission {submission.id} queued for review")
        
        # Trigger automated reviews
        await self._schedule_reviews(submission)
        
        return submission.id
    
    async def _schedule_reviews(self, submission: CodeSubmission):
        """Schedule appropriate reviewers based on code type"""
        review_tasks = []
        
        # Always include code review
        review_tasks.append(self._automated_code_review(submission))
        
        # Add performance validation for kernel code
        if any(lang in submission.language.lower() for lang in ['cpp', 'c++', 'hip', 'cuda']):
            review_tasks.append(self._performance_validation(submission))
        
        # Add security audit for all code
        review_tasks.append(self._security_audit(submission))
        
        # Add integration testing
        if submission.tests_included:
            review_tasks.append(self._integration_test(submission))
        
        # Execute reviews in parallel
        await asyncio.gather(*review_tasks, return_exceptions=True)
    
    async def _automated_code_review(self, submission: CodeSubmission) -> ReviewResult:
        """Automated code review using static analysis"""
        issues = []
        suggestions = []
        score = 10.0
        
        # Check for common issues
        code_lines = submission.code.split('\n')
        
        # Basic code quality checks
        if len([line for line in code_lines if line.strip()]) < 10:
            issues.append("Code submission too short for meaningful review")
            score -= 2.0
        
        # Check for TODO/FIXME comments
        todo_count = sum(1 for line in code_lines if 'TODO' in line or 'FIXME' in line)
        if todo_count > 0:
            suggestions.append(f"Found {todo_count} TODO/FIXME comments - consider addressing")
            score -= 0.5 * todo_count
        
        # Language-specific checks
        if submission.language.lower() in ['cpp', 'c++']:
            score = await self._cpp_specific_review(submission, score, issues, suggestions)
        elif submission.language.lower() == 'python':
            score = await self._python_specific_review(submission, score, issues, suggestions)
        
        status = ReviewStatus.APPROVED if score >= self.approval_threshold else ReviewStatus.NEEDS_CHANGES
        if issues:
            status = ReviewStatus.NEEDS_CHANGES
        
        return ReviewResult(
            reviewer_id="automated_code_reviewer",
            review_type=ReviewType.CODE_REVIEW,
            status=status,
            score=max(0, score),
            feedback=f"Automated review completed. Score: {score:.1f}/10",
            suggestions=suggestions,
            blocking_issues=issues
        )
    
    async def _cpp_specific_review(self, submission: CodeSubmission, score: float, issues: List[str], suggestions: List[str]) -> float:
        """C++ specific code review checks"""
        code = submission.code.lower()
        
        # Check for memory management
        if 'malloc' in code and 'free' not in code:
            issues.append("malloc() found without corresponding free()")
            score -= 3.0
        
        if 'new' in code and 'delete' not in code:
            suggestions.append("Consider using smart pointers or ensure proper delete calls")
            score -= 1.0
        
        # Check for GPU-specific patterns
        if 'hip' in code or 'cuda' in code:
            if '__global__' in code and '__syncthreads' not in code:
                suggestions.append("Consider adding __syncthreads() for thread synchronization")
            
            if 'shared' in code and '__syncthreads' not in code:
                suggestions.append("Shared memory usage should typically include __syncthreads()")
                score -= 0.5
        
        # Check for performance patterns
        if '#pragma unroll' not in code and 'for' in code:
            suggestions.append("Consider #pragma unroll for small, fixed loops")
        
        return score
    
    async def _python_specific_review(self, submission: CodeSubmission, score: float, issues: List[str], suggestions: List[str]) -> float:
        """Python specific code review checks"""
        code = submission.code
        
        # Check for imports
        if 'import torch' in code:
            if 'torch.cuda.is_available()' not in code and 'device' in code:
                suggestions.append("Consider checking CUDA availability before device operations")
        
        # Check for type hints
        if 'def ' in code and '->' not in code:
            suggestions.append("Consider adding type hints for better code documentation")
            score -= 0.5
        
        # Check for docstrings
        functions = [line for line in code.split('\n') if line.strip().startswith('def ')]
        if functions and '"""' not in code:
            suggestions.append("Consider adding docstrings to functions")
            score -= 0.5
        
        return score
    
    async def _performance_validation(self, submission: CodeSubmission) -> ReviewResult:
        """Validate performance characteristics"""
        score = 8.0  # Start with good score for perf
        issues = []
        suggestions = []
        
        # Check if performance data is provided
        if not submission.performance_data:
            suggestions.append("No performance data provided - consider adding benchmarks")
            score -= 1.0
        else:
            perf_data = submission.performance_data
            
            # Check for performance improvements
            if 'baseline_time' in perf_data and 'optimized_time' in perf_data:
                improvement = (perf_data['baseline_time'] - perf_data['optimized_time']) / perf_data['baseline_time']
                if improvement < 0:
                    issues.append(f"Performance regression detected: {improvement:.1%} slower")
                    score -= 5.0
                elif improvement > 0.1:
                    suggestions.append(f"Good performance improvement: {improvement:.1%} faster")
                    score += 1.0
            
            # Check memory usage
            if 'memory_usage' in perf_data:
                memory_mb = perf_data['memory_usage']
                if memory_mb > 1000:  # > 1GB
                    suggestions.append(f"High memory usage: {memory_mb}MB - consider optimization")
                    score -= 1.0
        
        status = ReviewStatus.APPROVED if score >= self.approval_threshold else ReviewStatus.NEEDS_CHANGES
        
        return ReviewResult(
            reviewer_id="performance_validator",
            review_type=ReviewType.PERFORMANCE_VALIDATION,
            status=status,
            score=max(0, score),
            feedback=f"Performance validation completed. Score: {score:.1f}/10",
            suggestions=suggestions,
            blocking_issues=issues
        )
    
    async def _security_audit(self, submission: CodeSubmission) -> ReviewResult:
        """Security audit for code submission"""
        score = 9.0
        issues = []
        suggestions = []
        
        code_lower = submission.code.lower()
        
        # Check for potential security issues
        dangerous_patterns = [
            ('system(', 'Potential command injection risk'),
            ('exec(', 'Dynamic code execution risk'),
            ('eval(', 'Dynamic code evaluation risk'),
            ('os.system', 'System command execution risk'),
            ('subprocess.call', 'Consider using subprocess.run with shell=False'),
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in code_lower:
                issues.append(warning)
                score -= 2.0
        
        # Check for input validation
        if 'input(' in code_lower and 'validate' not in code_lower:
            suggestions.append("Consider adding input validation")
            score -= 0.5
        
        # Check for hardcoded credentials (basic check)
        if any(word in code_lower for word in ['password', 'secret', 'token', 'key']) and '=' in submission.code:
            issues.append("Potential hardcoded credentials detected")
            score -= 3.0
        
        status = ReviewStatus.APPROVED if score >= self.approval_threshold and not issues else ReviewStatus.NEEDS_CHANGES
        
        return ReviewResult(
            reviewer_id="security_auditor",
            review_type=ReviewType.SECURITY_AUDIT,
            status=status,
            score=max(0, score),
            feedback=f"Security audit completed. Score: {score:.1f}/10",
            suggestions=suggestions,
            blocking_issues=issues
        )
    
    async def _integration_test(self, submission: CodeSubmission) -> ReviewResult:
        """Run integration tests"""
        score = 8.0
        issues = []
        suggestions = []
        
        # Create temporary file and try to compile/run basic checks
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_file_extension(submission.language), delete=False) as tmp_file:
                tmp_file.write(submission.code)
                tmp_file_path = tmp_file.name
            
            # Language-specific testing
            if submission.language.lower() in ['cpp', 'c++']:
                success = await self._test_cpp_compilation(tmp_file_path)
                if not success:
                    issues.append("Code does not compile")
                    score -= 5.0
            
            elif submission.language.lower() == 'python':
                success = await self._test_python_syntax(tmp_file_path)
                if not success:
                    issues.append("Python syntax errors detected")
                    score -= 5.0
        
        except Exception as e:
            issues.append(f"Testing error: {str(e)}")
            score -= 3.0
        
        finally:
            # Cleanup
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        
        status = ReviewStatus.APPROVED if score >= self.approval_threshold else ReviewStatus.NEEDS_CHANGES
        
        return ReviewResult(
            reviewer_id="integration_tester",
            review_type=ReviewType.INTEGRATION_TEST,
            status=status,
            score=max(0, score),
            feedback=f"Integration testing completed. Score: {score:.1f}/10",
            suggestions=suggestions,
            blocking_issues=issues
        )
    
    def _get_file_extension(self, language: str) -> str:
        """Get appropriate file extension for language"""
        ext_map = {
            'cpp': '.cpp',
            'c++': '.cpp', 
            'c': '.c',
            'python': '.py',
            'hip': '.hip',
            'cuda': '.cu'
        }
        return ext_map.get(language.lower(), '.txt')
    
    async def _test_cpp_compilation(self, file_path: str) -> bool:
        """Test C++ compilation"""
        try:
            result = subprocess.run(
                ['g++', '-fsyntax-only', file_path],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False
    
    async def _test_python_syntax(self, file_path: str) -> bool:
        """Test Python syntax"""
        try:
            result = subprocess.run(
                ['python3', '-m', 'py_compile', file_path],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False
    
    def get_review_summary(self, submission_id: str) -> Dict[str, Any]:
        """Get comprehensive review summary"""
        if submission_id not in self.reviews:
            return {"error": "Submission not found"}
        
        reviews = self.reviews[submission_id]
        if not reviews:
            return {"status": "pending", "message": "Reviews in progress"}
        
        # Calculate overall approval
        approved_count = sum(1 for r in reviews if r.status == ReviewStatus.APPROVED)
        average_score = sum(r.score for r in reviews) / len(reviews)
        
        all_blocking_issues = []
        all_suggestions = []
        
        for review in reviews:
            all_blocking_issues.extend(review.blocking_issues)
            all_suggestions.extend(review.suggestions)
        
        overall_status = "approved" if (approved_count >= self.required_reviewers and 
                                      average_score >= self.approval_threshold and 
                                      not all_blocking_issues) else "needs_changes"
        
        return {
            "submission_id": submission_id,
            "overall_status": overall_status,
            "average_score": average_score,
            "reviews_completed": len(reviews),
            "approved_reviews": approved_count,
            "blocking_issues": all_blocking_issues,
            "suggestions": list(set(all_suggestions)),  # Remove duplicates
            "detailed_reviews": [
                {
                    "reviewer": r.reviewer_id,
                    "type": r.review_type.value,
                    "status": r.status.value,
                    "score": r.score,
                    "feedback": r.feedback
                } for r in reviews
            ]
        }
    
    def add_review_result(self, submission_id: str, review: ReviewResult):
        """Add a review result (for external reviewers)"""
        if submission_id in self.reviews:
            self.reviews[submission_id].append(review)

# Integration with the main coordinator
class ReviewIntegratedCoordinator:
    """Extended coordinator with integrated quality control"""
    
    def __init__(self, coordinator, quality_controller):
        self.coordinator = coordinator
        self.qc = quality_controller
    
    async def submit_agent_work_for_review(self, task_id: str) -> str:
        """Submit completed agent work for review"""
        task = self.coordinator.get_task_status(task_id)
        if not task or task.status.value != 'completed':
            return "Task not completed or not found"
        
        # Extract code from task result
        if 'code' in task.result.get('response', {}):
            submission = CodeSubmission(
                id=f"review_{task_id}",
                task_id=task_id,
                agent_id=task.assigned_agent,
                code=task.result['response']['code'],
                language=self._detect_language(task.result['response']['code']),
                description=task.context.get('objective', 'Agent-generated code'),
                files_modified=task.context.get('files', []),
                tests_included='test' in task.result.get('response', {}),
                performance_data=task.result.get('response', {}).get('performance_notes')
            )
            
            review_id = await self.qc.submit_for_review(submission)
            return f"Submitted for review: {review_id}"
        
        return "No code found in task result"
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection"""
        if '#include' in code or 'hip' in code.lower():
            return 'cpp'
        elif 'import' in code and 'def ' in code:
            return 'python'
        else:
            return 'unknown'

if __name__ == "__main__":
    print("Quality Control System for Distributed AI Development")
    print("Multi-agent code review and validation ready!")