import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import oneLight from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-light";
import oneDark from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-dark";
import javascript from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/hljs/typescript";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import bash from "react-syntax-highlighter/dist/esm/languages/hljs/bash";
import json from "react-syntax-highlighter/dist/esm/languages/hljs/json";
import yaml from "react-syntax-highlighter/dist/esm/languages/hljs/yaml";
import sql from "react-syntax-highlighter/dist/esm/languages/hljs/sql";
import xml from "react-syntax-highlighter/dist/esm/languages/hljs/xml";
import { CitationList } from "@/components/CitationList";
import { cn } from "@/lib/utils";
import type { ChatMessage as ChatMessageData } from "@/hooks/useConversation";

SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("py", python);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("html", xml);
SyntaxHighlighter.registerLanguage("xml", xml);

interface ChatMessageProps {
  message: ChatMessageData;
  isDark: boolean;
  streaming?: boolean;
}

export function ChatMessage({ message, isDark, streaming = false }: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-secondary px-4 py-2.5 text-sm leading-relaxed text-foreground">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex w-full justify-start">
      <div className="w-full max-w-3xl">
        <div
          className={cn(
            "prose-editorial",
            streaming && "token-fade",
            message.blocked && "text-muted-foreground italic",
          )}
        >
          {message.content ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className ?? "");
                  const codeString = String(children).replace(/\n$/, "");
                  // Inline code (no language fence) — render plainly
                  // react-markdown 9 doesn't pass `inline`, so detect by absence of newline + no language
                  const isInline = !match && !codeString.includes("\n");
                  if (isInline) {
                    return (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                  return (
                    <SyntaxHighlighter
                      language={match ? match[1] : "text"}
                      style={isDark ? oneDark : oneLight}
                      PreTag="div"
                      customStyle={{
                        margin: 0,
                        padding: "0.9rem 1rem",
                        fontSize: "0.85rem",
                        background: "transparent",
                        borderRadius: 0,
                      }}
                      codeTagProps={{ style: { fontFamily: "var(--mono, monospace)" } }}
                    >
                      {codeString}
                    </SyntaxHighlighter>
                  );
                },
                a({ href, children, ...rest }) {
                  return (
                    <a href={href} target="_blank" rel="noreferrer" {...rest}>
                      {children}
                    </a>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          ) : (
            <span className="inline-block h-4 w-2 animate-pulse bg-muted-foreground/50 align-middle" />
          )}
        </div>
        {message.citations && message.citations.length > 0 && (
          <CitationList citations={message.citations} />
        )}
      </div>
    </div>
  );
}
