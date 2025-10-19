interface IButton {
  title: string | undefined;
  onClick: any;
  className: string;
}

export default function Button({ title, onClick, className }: IButton) {
  return (
    <button
      className={`bg-teal-500 text-white font-semibold px-6 py-3 rounded-4xl shadow-lg hover:bg-teal-600 transition ${className}`}
      onClick={onClick}
    >
      {title}
    </button>
  );
}
