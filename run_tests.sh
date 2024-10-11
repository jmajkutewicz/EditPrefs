#!/usr/bin/env bash

tests=(
    "collect_npov_diffs__consumer_test"
    "collect_retained_diffs_test"
    "comments_test"
    "diff_creation_test"
    "diff_saver_test"
    "filtering_test"
    "find_npov_edits_test"
    "generate_prompts_gpt3_test"
    "markup_processing_test"
    "sentence_tokenization_test"
    "template_rendering_test"
)
failed_tests=()

passed=0
failed=0
start_time=$(date +%s)
for t in "${tests[@]}"; do
  echo "Running test ${t}"

  if python -m unittest "test/${t}.py" -v; then
    ((passed++))
  else
    ((failed++))
    failed_tests+=("$t")
  fi
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "-------------------"
echo "Test Run Summary"
echo "Total tests run: $((passed + failed))"
echo "Passed: $passed"
echo "Failed: $failed"
for t in "${failed_tests[@]}"; do
  echo -e "\t${t}"
done
echo "Total duration: ${total_duration} seconds"