import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";

const Card = ({ title, children }) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-4 m-2">
      <h2 className="text-lg font-bold mb-2">{title}</h2>
      <div>{children}</div>
    </div>
  );
};

export default Card;