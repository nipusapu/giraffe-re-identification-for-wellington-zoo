import Button from "./button";

interface IDescription {
  title: String;
  description: String;
}

export default function Description({ title, description }: IDescription) {
  return (
    <div className="text-2xl text-white ">
      <div className="pt-2">{title}</div>
      <div className="flex justify-center items-center mt-3 p-10 font-bold leading-10">
        <p>{description}</p>
      </div>
    </div>
  );
}
