import { useState, type KeyboardEvent } from 'react';
import { X, Sparkles } from 'lucide-react';

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  onSuggest?: () => void;
  isSuggesting?: boolean;
}

export function TagInput({
  tags,
  onChange,
  placeholder = 'Добавьте тег и нажмите Enter...',
  onSuggest,
  isSuggesting = false
}: TagInputProps) {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      const newTag = inputValue.trim();
      if (!tags.includes(newTag)) {
        onChange([...tags, newTag]);
      }
      setInputValue('');
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      onChange(tags.slice(0, -1));
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(tags.filter(tag => tag !== tagToRemove));
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5 min-h-[2.5rem] p-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500/30 transition-all duration-150">
        {tags.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-700/50 text-slate-200 text-[13px] rounded border border-slate-600/30"
          >
            {tag}
            <button
              type="button"
              onClick={() => removeTag(tag)}
              className="p-0.5 hover:bg-slate-600/50 rounded transition-colors duration-150"
            >
              <X className="w-3 h-3 text-slate-400" />
            </button>
          </span>
        ))}
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={tags.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[120px] bg-transparent text-slate-200 placeholder-slate-500 outline-none text-[13px]"
        />
      </div>
      {onSuggest && (
        <button
          type="button"
          onClick={onSuggest}
          disabled={isSuggesting}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-blue-400 bg-blue-500/8 border border-blue-500/15 rounded-lg hover:bg-blue-500/15 hover:border-blue-500/25 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Sparkles className={`w-3.5 h-3.5 ${isSuggesting ? 'animate-pulse' : ''}`} />
          {isSuggesting ? 'Подбираем...' : 'Подобрать с помощью ИИ'}
        </button>
      )}
    </div>
  );
}
