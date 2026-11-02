// Knowledge-gap rows for the Analytics page. Newer servers send `gap_queries`
// (judged on each question's latest search: no results, or best rerank score below
// the gap threshold); older ones only `zero_queries`.
export function gapRows(d) {
  if (!d) return []
  if (Array.isArray(d.gap_queries)) {
    return d.gap_queries.map((g) => ({
      query: g.query,
      count: g.count,
      score: g.result_count === 0 || g.top_score == null ? (g.result_count === 0 ? 'no results' : '—') : g.top_score.toFixed(2),
    }))
  }
  return (d.zero_queries || []).map((g) => ({ query: g.query, count: g.count, score: 'no results' }))
}

export function gapDesc(d) {
  if (d && d.gap_threshold != null) {
    return `Questions whose latest search found nothing relevant (no results, or best rerank score below ${d.gap_threshold}) — candidates for new content`
  }
  return 'Searches that returned nothing — candidates for new content'
}
