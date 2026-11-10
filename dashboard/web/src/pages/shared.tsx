import { ArrowRight } from 'lucide-react'
import { Button, Empty, PageHead } from '../components/ui'
import { useStore } from '../lib/store'

/** Empty state for pages that describe a prediction. */
export function NeedsResult({ title }: { title: string }) {
  const { go, runExample, busy } = useStore()
  return (
    <div>
      <PageHead title={title} />
      <Empty title="Nothing to show yet" action={
        <div className="mt-2 flex flex-wrap gap-3">
          <Button variant="primary" onClick={() => go('workspace')}>Open the workspace <ArrowRight size={16} /></Button>
          <Button onClick={runExample} disabled={!!busy}>Run the Atlanta example</Button>
        </div>}>
        This page describes a prediction. Run one in the workspace, or start with the recorded Atlanta example.
      </Empty>
    </div>
  )
}
