type Props = { text: string };
export default function StepBubble({ text }: Props) {
  return (
    <div className="mb-2 max-w-2xl">
      <div className="bg-gray-50 border-l-4 border-[color:var(--accent)]
                      px-3 py-2 text-sm text-gray-700 shadow-sm rounded-r">
        {text}
      </div>
    </div>
  );
} 