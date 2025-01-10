/**
 * k6 load test for quorabust-serve /predict.
 *
 * Prereq: server running with a loaded model, e.g.
 *   export QUORABUST_MODEL_PATH=models/your.pkl
 *   quorabust-serve --port 8000
 *
 * Run:
 *   k6 run -e BASE_URL=http://127.0.0.1:8000 loadtests/k6_predict.js
 *
 * Tune thresholds to your SLO (p95 latency, error rate).
 */
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 5,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<800"],
  },
};

const BASE = __ENV.BASE_URL || "http://127.0.0.1:8000";

export default function () {
  const payload = JSON.stringify({
    question1: ["how to learn python programming"],
    question2: ["best resources to learn python"],
  });
  const res = http.post(`${BASE}/predict`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(res, {
    "predict status 200": (r) => r.status === 200,
    "has proba_duplicate": (r) => {
      try {
        const b = JSON.parse(r.body);
        return Array.isArray(b.proba_duplicate) && b.proba_duplicate.length === 1;
      } catch {
        return false;
      }
    },
  });
  sleep(0.05);
}
