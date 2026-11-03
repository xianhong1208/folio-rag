// The terminal caption every page opens with (the Overview draws its own, with the pipeline).
// `key` restarts the typing animation on each navigation.
export default function PageCap({ title, query }) {
  return (
    <div className="term-cap page-cap">
      <h1>{title}</h1>
      <div className="term-q" key={title + query}>
        &#10095; <span className="typed" style={{ '--chars': query.length }}>{query}</span><span className="cursor blink">&#9613;</span>
      </div>
    </div>
  )
}
