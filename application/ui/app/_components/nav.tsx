import Image from 'next/image';
import { AiOutlineMenu } from 'react-icons/ai';

export default function Nav() {
  return (
    <div className="flex justify-between">
      <div className="w-2xl p-4">
        <AiOutlineMenu size={50} />
      </div>
      <div className="p-4">
        <Image src="/images/zoolog.png" alt="logo" width="100" height="100" />
      </div>
    </div>
  );
}
