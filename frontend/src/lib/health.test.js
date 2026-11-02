import { describe, it, expect } from 'vitest'
import { serviceStates } from './health'

describe('serviceStates', () => {
  it('maps health checks to overview pills', () => {
    expect(serviceStates([
      { name: 'Database', status: 'ok' },
      { name: 'Embedding', status: 'ok' },
      { name: 'LLM', status: 'down' },
      { name: 'Reranker', status: 'disabled' },
    ])).toEqual({ embed: 'online', llm: 'offline', rerank: 'disabled' })
  })
  it('treats unknown statuses as offline and tolerates no data', () => {
    expect(serviceStates([{ name: 'LLM', status: '???' }, { name: 'Embedding', status: 'unset' }]))
      .toEqual({ llm: 'offline', embed: 'unset' })
    expect(serviceStates(null)).toEqual({})
  })
})
