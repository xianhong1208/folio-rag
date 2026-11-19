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
  it('maps the document-parsing check to the vision pill', () => {
    expect(serviceStates([{ name: 'Document Parsing', status: 'unset' }])).toEqual({ vision: 'unset' })
    expect(serviceStates([{ name: 'Document Parsing', status: 'ok' }])).toEqual({ vision: 'online' })
  })
  it('treats unknown statuses as offline and tolerates no data', () => {
    expect(serviceStates([{ name: 'LLM', status: '???' }, { name: 'Embedding', status: 'unset' }]))
      .toEqual({ llm: 'offline', embed: 'unset' })
    expect(serviceStates(null)).toEqual({})
  })
})
